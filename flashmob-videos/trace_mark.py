"""東大のイチョウマーク画像から, 色ごとにきれいに塗り直した高解像度 PNG (透過) を作る.
assets/utokyo_mark.png (マークだけ) と assets/utokyo_logo.png (下に「東京大学」の文字付き) を書き出す.

使い方: python3 trace_mark.py 元画像.jpg [中心x 中心y 半径]
元画像内のマークの中心と半径 (px) は, 省略時は付属の画像 (355x244) に合わせた値.
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

YELLOW = (250, 189, 3)
BLUE = (24, 126, 196)
S = 1000  # 出力サイズ (マークの直径がほぼ S)


def trace(src, cx=176.5, cy=91.0, r=75.0, out="assets/utokyo_mark.png"):
    im = Image.open(src).convert("RGB")
    box = tuple(int(round(v)) for v in (cx - r * 1.02, cy - r * 1.02, cx + r * 1.02, cy + r * 1.02))
    big = np.asarray(im.crop(box).resize((S * 2, S * 2), Image.BICUBIC)).astype(float)
    refs = [np.array(c, float) for c in ((255, 255, 255), YELLOW, BLUE)]
    label = np.stack([np.linalg.norm(big - c, axis=2) for c in refs]).argmin(0)
    rgba = np.zeros((S * 2, S * 2, 4), np.uint8)
    for k, col in ((1, YELLOW), (2, BLUE)):
        m = Image.fromarray(((label == k) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
        m = np.asarray(m) > 127  # ぼかしてからしきい値 → 拡大のギザギザが消える
        rgba[m, :3] = col
        rgba[m, 3] = 255
    mark = Image.fromarray(rgba).resize((S, S), Image.LANCZOS)
    mark.save(out)
    print("wrote", out)

    # 文字部分: マークの下の黒い文字を拡大し, ぼかし → コントラストを上げてくっきりさせる
    k = S / (2 * r * 1.02)  # 元画像 1px → 出力 px
    g = np.asarray(im.convert("L"))
    rows = np.nonzero((g[int(cy + r):] < 128).any(axis=1))[0] + int(cy + r)
    ty0, ty1 = rows.min() - 2, rows.max() + 3
    tbox = (int(cx - r * 1.02), ty0, int(cx + r * 1.02), ty1)
    tw, th = int((tbox[2] - tbox[0]) * k), int((ty1 - ty0) * k)
    t = im.convert("L").crop(tbox).resize((tw * 2, th * 2), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))
    t = t.point(lambda v: 0 if v < 95 else 255 if v > 165 else int((v - 95) * 255 / 70)).resize((tw, th), Image.LANCZOS)
    text_rgba = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    text_rgba.putalpha(t.point(lambda v: 255 - v))
    gap = int((ty0 - (cy + r)) * k)
    logo = Image.new("RGBA", (S, S + gap + th), (0, 0, 0, 0))
    logo.paste(mark, (0, 0), mark)
    logo.paste(text_rgba, ((S - tw) // 2, S + gap), text_rgba)
    logo_out = out.replace("utokyo_mark", "utokyo_logo")
    logo.save(logo_out)
    print("wrote", logo_out)


if __name__ == "__main__":
    args = sys.argv[1:]
    trace(args[0], *map(float, args[1:4]))
