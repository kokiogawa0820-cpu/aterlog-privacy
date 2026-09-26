"""共通ユーティリティ: 描画・字幕・ツッコミ演出・効果音合成・動画書き出し."""
import math
import os
import subprocess
import wave

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080  # 横長 16:9
FPS = 30
SR = 44100

FONT_PATHS = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]
BG = (247, 243, 234)
INK = (40, 36, 32)

_font_cache = {}


def font(size):
    size = int(size)
    if size not in _font_cache:
        path = next(p for p in FONT_PATHS if os.path.exists(p))
        _font_cache[size] = ImageFont.truetype(path, size)
    return _font_cache[size]


def clamp01(x):
    return max(0.0, min(1.0, x))


def ease(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out_back(t, s=1.9):
    t = clamp01(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def text(d, xy, s, size, fill=INK, stroke=0, stroke_fill=(255, 255, 255), anchor="mm"):
    d.multiline_text(xy, s, font=font(size), fill=fill, anchor=anchor, align="center",
                     stroke_width=stroke, stroke_fill=stroke_fill, spacing=size * 0.25)


def caption(d, s, x=W / 2, y=H - 120, size=64, sub=None, pop=1.0, width=None):
    """テロップ. (x, y) が中心, width は枠の最小幅. pop は 0→1 の出現アニメ用."""
    if not s:
        return
    size = size * (0.6 + 0.4 * ease_out_back(pop))
    f = font(size)
    box = d.multiline_textbbox((x, y), s, font=f, anchor="mm", align="center", spacing=size * 0.25)
    pad = 34
    half = max((width or 0) / 2, (box[2] - box[0]) / 2 + pad)
    d.rounded_rectangle((x - half, box[1] - pad, x + half, box[3] + pad),
                        radius=28, fill=(255, 255, 255), outline=INK, width=5)
    text(d, (x, y), s, size)
    if sub:
        text(d, (x, box[3] + pad + 50), sub, 42, fill=(110, 100, 90))


def header(d, s):
    d.rectangle((0, 0, W, 110), fill=INK)
    text(d, (W / 2, 58), s, 54, fill=(255, 255, 255))


def tsukkomi(img, u, word="無理やん！", base_size=175):
    """u: 0→1. 背景を暗くし, 集中線 + ドーンと文字を出す."""
    t = u * 2.5  # 秒換算 (シーン長 2.5s 前提の目安)
    dark = Image.new("RGB", img.size, (0, 0, 0))
    img.paste(Image.blend(img, dark, 0.55 * clamp01(t / 0.12)))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(int(t * 12))  # 集中線はコマ撮り風に12fpsで揺らす
    cx, cy = W / 2, H * 0.5
    for k in range(90):
        a = 2 * math.pi * k / 90 + rng.uniform(-0.02, 0.02)
        r0 = rng.uniform(330, 520)
        w = rng.uniform(0.006, 0.02)
        pts = [(cx + math.cos(a - w) * 2000, cy + math.sin(a - w) * 2000),
               (cx + math.cos(a) * r0, cy + math.sin(a) * r0),
               (cx + math.cos(a + w) * 2000, cy + math.sin(a + w) * 2000)]
        d.polygon(pts, fill=(255, 255, 255))
    scale = lerp(3.2, 1.0, ease(t / 0.14)) if t < 0.14 else 1.0
    shake = max(0.0, 1 - t / 0.8)
    dx = rng.uniform(-1, 1) * 28 * shake
    dy = rng.uniform(-1, 1) * 28 * shake
    tilt = -6
    size = int(base_size * scale)
    layer = Image.new("RGBA", (W * 2, 900), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.multiline_text((W, 450), word, font=font(size), anchor="mm", align="center", fill=(255, 238, 60),
            stroke_width=max(6, size // 11), stroke_fill=(210, 20, 30))
    layer = layer.rotate(tilt, resample=Image.BICUBIC)
    img.paste(layer, (int(-W / 2 + dx), int(cy - 450 + dy)), layer)


# ---------------------------------------------------------------- 効果音
class Audio:
    def __init__(self, dur):
        self.buf = np.zeros(int(dur * SR) + SR)

    def add(self, t, sig, gain=1.0):
        i = int(t * SR)
        n = min(len(sig), len(self.buf) - i)
        if n > 0:
            self.buf[i:i + n] += sig[:n] * gain

    @staticmethod
    def env(n, attack=0.005, decay=None):
        tt = np.arange(n) / SR
        e = np.minimum(1, tt / attack)
        if decay:
            e *= np.exp(-tt / decay)
        return e

    def pop(self, t, freq=660, gain=0.45):
        n = int(0.12 * SR)
        tt = np.arange(n) / SR
        f = freq * (1 + 1.5 * np.exp(-tt / 0.02))
        self.add(t, np.sin(2 * np.pi * np.cumsum(f) / SR) * self.env(n, decay=0.04), gain)

    def tick(self, t, freq=1800, gain=0.25):
        n = int(0.04 * SR)
        tt = np.arange(n) / SR
        self.add(t, np.sin(2 * np.pi * freq * tt) * self.env(n, 0.001, 0.008), gain)

    def whoosh(self, t, dur=0.5, gain=0.35):
        n = int(dur * SR)
        noise = np.random.default_rng(1).normal(size=n)
        k = np.hanning(n)
        sm = np.convolve(noise, np.ones(20) / 20, mode="same")
        self.add(t, sm * k * 3, gain)

    def sweep(self, t, f0, f1, dur, gain=0.3, wobble=0.0):
        n = int(dur * SR)
        tt = np.arange(n) / SR
        f = np.geomspace(f0, f1, n) * (1 + wobble * np.sin(2 * np.pi * 14 * tt))
        self.add(t, np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(n) ** 0.3, gain)

    def crack(self, t, gain=0.5):
        n = int(0.05 * SR)
        noise = np.random.default_rng(int(t * 1000)).normal(size=n)
        self.add(t, noise * self.env(n, 0.0005, 0.006), gain)

    def chime(self, t, gain=0.3):
        for k, f in enumerate([784, 988, 1175, 1568]):
            n = int(0.8 * SR)
            tt = np.arange(n) / SR
            self.add(t + k * 0.09, np.sin(2 * np.pi * f * tt) * self.env(n, 0.003, 0.25), gain)

    def don(self, t, gain=0.9):
        n = int(0.9 * SR)
        tt = np.arange(n) / SR
        f = 55 + 140 * np.exp(-tt / 0.05)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * self.env(n, 0.002, 0.3)
        hit = np.random.default_rng(7).normal(size=n) * self.env(n, 0.001, 0.03)
        self.add(t, body + 0.5 * hit, gain)

    def write(self, path):
        b = self.buf / max(1.0, np.abs(self.buf).max() / 0.9)
        pcm = (b * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- 書き出し
def render(scenes, out_path, audio=None, preview_times=()):
    """scenes: [(秒数, fn(img, draw, 経過秒, 進捗0-1))]. preview_times の秒でPNGも保存."""
    total = sum(s[0] for s in scenes)
    tmp = out_path + ".noaudio.mp4"
    writer = imageio.get_writer(tmp, fps=FPS, codec="libx264", macro_block_size=1,
                                ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p", "-preset", "medium"])
    previews = sorted(preview_times)
    frame = 0
    t0 = 0.0
    for dur, fn in scenes:
        n = int(round(dur * FPS))
        for i in range(n):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            fn(img, d, i / FPS, i / max(1, n - 1))
            gt = frame / FPS
            while previews and previews[0] <= gt:
                img.save(f"{os.path.splitext(out_path)[0]}_preview_{previews.pop(0):05.1f}s.png")
            writer.append_data(np.asarray(img))
            frame += 1
        t0 += dur
    writer.close()
    if audio is None:
        os.replace(tmp, out_path)
        return total
    wav = out_path + ".wav"
    audio.write(wav)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path], check=True)
    os.remove(tmp)
    os.remove(wav)
    return total
