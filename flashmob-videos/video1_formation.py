"""動画1: ◯(=人)を紹介しながら配置 → どんどん加速して150人 → (ツッコミ「参列者全員参加させるんか！」は撮影側で)

--tsukkomi を付けると最後に文字のツッコミ演出も入れる.
"""
import colorsys
import math
import sys

import numpy as np

from common import (INK, W, Audio, caption, clamp01, ease, ease_out_back, font, header, mix, render, text,
                    tsukkomi)

TITLE = "フラッシュモブ振付会議"
TOTAL = 150

# ---- 名前と肩書き (仮). 名前が決まったらここを書き換えてください -------------------
# 先頭4人はゆっくり紹介, 5人目以降は加速しながら紹介される.
NAMED = [
    ("田中", "新郎の友人代表"),
    ("佐藤", "新郎の大学の同期"),
    ("鈴木", "新婦の幼なじみ"),
    ("高橋", "言い出しっぺ"),
    ("伊藤", "新郎の会社の後輩"),
    ("渡辺", "新婦の職場の先輩"),
    ("山本", "新郎の上司"),
    ("中村", "新婦の上司"),
    ("小林", "新郎のいとこ"),
    ("加藤", "新婦のおじ"),
]
# 名前のない人は, 何人目かに応じてこの肩書きがテロップに出る (境目の人数, 肩書き)
CROWD_ROLES = [
    (14, "新郎の親戚"),
    (22, "新婦の親戚"),
    (32, "新郎の会社の人たち"),
    (45, "新婦の会社の人たち"),
    (60, "新郎新婦の恩師"),
    (72, "受付の人"),
    (82, "司会者"),
    (90, "カメラマン"),
    (104, "新郎のおばあちゃん"),
    (118, "新婦のおばあちゃん"),
    (135, "ご近所の方"),
    (TOTAL, "その他の参列者のみなさん"),
]
SLOW = 4  # ゆっくり紹介する人数
# ---------------------------------------------------------------------------------

CX, CY = 540, 890
SPACING = 0.25  # ワールド座標での間隔 (ヒマワリ配置)
RW = 0.1  # ワールド座標での◯の半径
UNIT = 400  # ズーム1.0のとき 1.0 = 400px
GOLDEN = math.pi * (3 - math.sqrt(5))
STAGE = (60, 420, 1020, 1360)
FLOOR = (233, 224, 206)

POS = np.array([[SPACING * math.sqrt(k + 0.3) * math.cos(k * GOLDEN),
                 SPACING * math.sqrt(k + 0.3) * math.sin(k * GOLDEN) * 0.95] for k in range(TOTAL)])
COLORS = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb((k * 0.618) % 1, 0.62, 0.9)) for k in range(TOTAL)]


def role_of(k):
    if k < len(NAMED):
        return NAMED[k]
    for limit, role in CROWD_ROLES:
        if k < limit:
            return None, role
    return None, CROWD_ROLES[-1][1]


def placement_times(t0):
    """各人が置かれる時刻. 最初の数人はゆっくり→名前ありの人は加速→あとは肩書きごとに一定時間で一気に."""
    times, t = [], t0
    for k in range(SLOW):
        times.append(t)
        t += 1.45
    dt = 0.62
    for k in range(SLOW, len(NAMED)):
        times.append(t)
        t += dt
        dt *= 0.82
    start = len(NAMED)
    for j, (limit, _) in enumerate(CROWD_ROLES):
        n = max(0, min(limit, TOTAL) - start)
        span = 0.55 - 0.15 * j / max(1, len(CROWD_ROLES) - 1)  # 肩書き1つあたりの表示時間
        times += [t + span * i / max(1, n) for i in range(n)]
        t += span
        start += n
    times += [t] * (TOTAL - len(times))
    return np.array(times[:TOTAL])


def zoom_for(n):
    radius = SPACING * math.sqrt(max(n, 1) + 0.3) + RW * 1.4
    return min(1.0, 1.08 / radius)


def to_px(p, z):
    return CX + p[0] * UNIT * z, CY + p[1] * UNIT * z


def draw_stage(d, z):
    d.rounded_rectangle(STAGE, radius=24, fill=FLOOR, outline=INK, width=4)
    step = 0.3 * UNIT * z
    k0 = int((STAGE[2] - CX) / step) + 1
    for k in range(-k0, k0 + 1):
        x = CX + k * step
        if STAGE[0] < x < STAGE[2]:
            d.line((x, STAGE[1] + 3, x, STAGE[3] - 3), fill=(218, 207, 186), width=2)
        y = CY + k * step
        if STAGE[1] < y < STAGE[3]:
            d.line((STAGE[0] + 3, y, STAGE[2] - 3, y), fill=(218, 207, 186), width=2)
    text(d, (CX, STAGE[3] + 45), "▼ 客席 ▼", 38, fill=(120, 110, 95))


def draw_counter(d, n, bump):
    x1, y1, x2, y2 = 640, 200, 1040, 360
    d.rounded_rectangle((x1, y1, x2, y2), radius=22, fill=(255, 255, 255), outline=INK, width=5)
    text(d, ((x1 + x2) / 2, y1 + 38), "配置人数", 38, fill=(110, 100, 90))
    size = 84 * (1 + 0.18 * bump)
    col = mix(INK, (210, 30, 40), clamp01((n - 40) / 110))
    text(d, ((x1 + x2) / 2, y1 + 105), f"{n}人", size, fill=col)


def draw_people(d, times, t, z):
    for k in range(TOTAL):
        age = t - times[k]
        if age < 0:
            break
        s = ease_out_back(age / 0.25)
        x, y = to_px(POS[k], z)
        r = RW * UNIT * z * s
        d.ellipse((x - r + 3, y - r + 5, x + r + 3, y + r + 5), fill=(200, 190, 170))
        d.ellipse((x - r, y - r, x + r, y + r), fill=COLORS[k], outline=INK, width=max(2, int(4 * z)))
        name = role_of(k)[0]
        if name and r > 12:
            text(d, (x, y), name, r * (0.55 if len(name) > 2 else 0.68), fill=(255, 255, 255),
                 stroke=max(1, int(3 * z)), stroke_fill=INK)


def build(out_path, with_tsukkomi=False, previews=()):
    audio = Audio(40)
    scenes = []
    t_title = 1.6
    times = placement_times(t_title)
    t_end_place = times[-1] + 0.4
    hold = 2.8

    def title(img, d, t, u):
        header(d, TITLE)
        draw_stage(d, 1.0)
        a = ease_out_back(t / 0.5)
        d.rounded_rectangle((110, 720, 970, 1060), radius=30, fill=(255, 255, 255), outline=INK, width=5)
        text(d, (W / 2, 890), "フォーメーション\n説明会", int(96 * (0.5 + 0.5 * a)))
        caption(d, "結婚式の余興です", sub="（新郎新婦にはナイショ）", pop=t / 0.3)

    scenes.append((t_title, title))
    audio.pop(0.05, 440)

    zoom = {"z": 1.0, "last_n": 0, "bump_t": -9.0}

    def placing(img, d, lt, u):
        t = t_title + lt
        n = int(np.sum(times <= t))
        if n != zoom["last_n"]:
            zoom["bump_t"] = t
            zoom["last_n"] = n
        zoom["z"] += (zoom_for(n) - zoom["z"]) * 0.18  # カメラはなめらかに引く
        z = zoom["z"]
        header(d, TITLE)
        draw_stage(d, z)
        draw_people(d, times, t, z)
        draw_counter(d, n, max(0.0, 1 - (t - zoom["bump_t"]) / 0.15))
        if n == 0:
            return
        k = n - 1
        name, role = role_of(k)
        if k < SLOW:
            x, y = to_px(POS[k], z)
            r = RW * UNIT * z
            d.line((x, y - r - 12, x, y - r - 60), fill=(210, 30, 40), width=6)
            d.polygon([(x - 14, y - r - 26), (x + 14, y - r - 26), (x, y - r - 6)], fill=(210, 30, 40))
            caption(d, f"これが{name}さん" if k == 0 else f"こちら{name}さん", sub=f"（{role}）",
                    pop=(t - times[k]) / 0.25)
        elif name:
            caption(d, f"{name}さん（{role}）", size=58)
        else:
            caption(d, role, size=62)

    scenes.append((t_end_place - t_title, placing))

    # 効果音: 最初は1人ずつポン, 速くなったらフレームごとにカチカチ (音程が上がっていく)
    last = -1.0
    for k, tk in enumerate(times):
        if tk - last >= 0.07:
            if k < 12:
                audio.pop(tk, 520 * 1.05 ** k)
            else:
                audio.tick(tk, 900 + 8 * k, 0.22)
            last = tk

    def final(img, d, t, u):
        z = zoom["z"] = zoom["z"] + (zoom_for(TOTAL) - zoom["z"]) * 0.18
        header(d, TITLE)
        draw_stage(d, z)
        draw_people(d, times, 999, z)
        draw_counter(d, TOTAL, 0)
        caption(d, "以上、150名で踊ります！", sub="（招待客リストより）", pop=t / 0.3)

    scenes.append((hold, final))
    audio.chime(t_end_place)

    if with_tsukkomi:
        def punch(img, d, t, u):
            final(img, d, hold, 1)
            tsukkomi(img, t / 2.5, "参列者全員\n参加させるんか！", base_size=120)

        scenes.append((2.6, punch))
        audio.don(t_end_place + hold)

    total = render(scenes, out_path, audio, preview_times=previews)
    print(f"wrote {out_path} ({total:.1f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "video1_formation.mp4"
    build(out, with_tsukkomi="--tsukkomi" in sys.argv,
          previews=[1.0, 3.5, 8.0, 9.5, 10.5, 11.5, 13.0, 15.0] if "--preview" in sys.argv else [])
