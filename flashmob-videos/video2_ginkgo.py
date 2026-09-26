"""動画2: 3体の人形が関節を無視した変形でイチョウマークを作る → 「無理やん！」

※ 東京大学の公式ロゴそのものではなく, 2枚のイチョウの葉を組み合わせたそれ風のシルエットです.
"""
import math
import sys

import numpy as np

from common import (INK, W, Audio, caption, clamp01, ease, ease_out_back, header, mix, render, text,
                    tsukkomi)

TITLE = "3人でイチョウマーク作ります"
WOOD = (224, 185, 138)
WOOD_D = (150, 105, 60)
LEAF = (126, 178, 216)  # 淡青っぽい色
NP = 13  # 各パーツのサンプル点数 (関節は 0, 6, 12 番)
LIMBS = ["legL", "armL", "torso", "armR", "legR"]


def chain(pts, n=NP):
    """関節 (3点) を通る折れ線を, 各区間 (n-1)/2 分割で n 点にする."""
    a, b, c = (np.array(p, float) for p in pts)
    h = (n - 1) // 2
    s = np.linspace(0, 1, h + 1)[:, None]
    return np.vstack([a + (b - a) * s, (b + (c - b) * s)[1:]])


def resample(poly, n):
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n)
    return np.column_stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])])


# ------------------------------------------------------------------ ポーズ
def pose_stand(cx, base, s=1.7, wave=0.0):
    p = lambda x, y: (cx + x * s, base + y * s)
    return {
        "torso": chain([p(0, -150), p(0, -75), p(0, 0)]),
        "armL": chain([p(-38, -140), p(-52 - 20 * wave, -80 - 40 * wave), p(-58 - 10 * wave, -18 - 110 * wave)]),
        "armR": chain([p(38, -140), p(52, -80), p(58, -18)]),
        "legL": chain([p(-18, 0), p(-24, 85), p(-28, 170)]),
        "legR": chain([p(18, 0), p(24, 85), p(28, 170)]),
        "head": (cx, base - 196 * s, 0.0),
    }


def pose_stretch(cx, base, s=1.7):
    """腕だけ3倍に伸びて真上へ."""
    q = pose_stand(cx, base, s)
    for side, sx in (("armL", -1), ("armR", 1)):
        sh = np.array([cx + 38 * s * sx, base - 140 * s])
        q[side] = chain([sh, sh + [sx * 25, -260], sh + [sx * 60, -520]])
    return q


def spiral(c, r0, r1, a0, turns, n=NP):
    a = a0 + np.linspace(0, 2 * math.pi * turns, n)
    r = np.linspace(r0, r1, n)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def pose_bent(cx, base, s=1.7, k=0):
    """関節が逆に曲がる: 腕はぐるぐる, 膝は逆くの字, 胴体はC字, 首は180度."""
    q = pose_stand(cx, base, s)
    neck = np.array([cx, base - 150 * s])
    a = np.linspace(-0.2, 1.4, NP) + k
    q["torso"] = np.column_stack([cx + 90 * np.sin(a * 2) - 40, np.linspace(neck[1], base, NP)])
    q["armL"] = spiral((cx - 150, base - 330), 120, 10, math.pi * 0.2 + k, 1.6)
    q["armR"] = spiral((cx + 150, base - 330), 120, 10, math.pi * 0.8 - k, -1.6)
    q["legL"] = chain([(cx - 30, base), (cx - 140, base + 90), (cx - 10, base + 280)])
    q["legR"] = chain([(cx + 30, base), (cx + 150, base + 60), (cx + 40, base + 290)])
    q["head"] = (cx + 20, base - 360, math.pi)
    return q


# ------------------------------------------------------------------ イチョウ
BX, BY = 540, 1180  # 2枚の葉の付け根
RAD = 440
SPREAD = math.radians(30)
TILT = math.radians(36)


def leaf_outline(axis, n=200):
    def pt(phi, r):
        return (BX + r * math.sin(phi), BY - r * math.cos(phi))

    def top_r(phi):
        d = phi - axis
        notch = 0.27 * math.exp(-(d / 0.09) ** 2)
        return RAD * (1 - notch + 0.018 * math.sin(14 * d) - 0.10 * (d / SPREAD) ** 4)

    left = [pt(axis - SPREAD, RAD * f) for f in np.linspace(0, 1, 30) ** 1.2]
    top = [pt(p, top_r(p)) for p in np.linspace(axis - SPREAD, axis + SPREAD, 120)]
    right = [pt(axis + SPREAD, RAD * f) for f in np.linspace(1, 0, 30) ** 1.2]
    # 付け根が少しくびれるよう左右の辺を内側にしならせる
    left = [(x + (BX - x) * 0.12 * math.sin(math.pi * k / 29), y) for k, (x, y) in enumerate(left)]
    right = [(x + (BX - x) * 0.12 * math.sin(math.pi * k / 29), y) for k, (x, y) in enumerate(right)]
    return resample(left + top + right, n)


LEAF_L = leaf_outline(-TILT)
LEAF_R = leaf_outline(TILT)


def notch_point(axis, f=0.7):
    return (BX + RAD * f * math.sin(axis), BY - RAD * f * math.cos(axis))


def pose_leaf_outline(outline, axis):
    pts = resample(outline, 5 * (NP - 1) + 1)
    q = {name: pts[k * (NP - 1): (k + 1) * (NP - 1) + 1] for k, name in enumerate(LIMBS)}
    hx, hy = notch_point(axis, 0.83)
    q["head"] = (hx, hy, axis)
    return q


def vein(axis, f0=0.08, f1=0.62):
    return resample([notch_point(axis, f0), notch_point(axis, f1)], NP)


def pose_leaf_veins():
    return {
        "torso": resample([(BX, BY), (BX + 25, BY + 150), (BX + 10, BY + 230)], NP),  # 茎
        "armL": vein(-TILT - 0.35),
        "legL": vein(-TILT + 0.3),
        "armR": vein(TILT + 0.35),
        "legR": vein(TILT - 0.3),
        "head": (BX + 10, BY + 262, 0.0),
    }


# ------------------------------------------------------------------ 描画
def interp_pose(a, b, u, lag=0.25):
    """パーツ先端ほど遅れて動く → ぐにゃっとした変形."""
    out = {}
    for name in LIMBS:
        delay = np.linspace(0, lag, NP)[:, None]
        uu = np.clip((u - delay) / (1 - lag), 0, 1)
        uu = uu * uu * (3 - 2 * uu)
        out[name] = a[name] + (b[name] - a[name]) * uu
    ha, hb = np.array(a["head"]), np.array(b["head"])
    e = ease(u)
    out["head"] = tuple(ha + (hb - ha) * e)
    return out


def draw_puppet(d, q, face="smile", width=26, extra_spin=0.0):
    for name in ["legL", "legR", "torso", "armL", "armR"]:
        pts = [tuple(p) for p in q[name]]
        d.line(pts, fill=WOOD_D, width=width + 8, joint="curve")
        for p in (pts[0], pts[-1]):
            d.ellipse((p[0] - (width + 8) / 2, p[1] - (width + 8) / 2, p[0] + (width + 8) / 2, p[1] + (width + 8) / 2),
                      fill=WOOD_D)
    for name in ["legL", "legR", "torso", "armL", "armR"]:
        pts = [tuple(p) for p in q[name]]
        d.line(pts, fill=WOOD, width=width, joint="curve")
        for p in (pts[0], pts[-1]):
            d.ellipse((p[0] - width / 2, p[1] - width / 2, p[0] + width / 2, p[1] + width / 2), fill=WOOD)
    for name in LIMBS:  # 関節の球
        for k in (0, NP // 2, NP - 1):
            x, y = q[name][k]
            r = width * 0.62
            d.ellipse((x - r, y - r, x + r, y + r), fill=(236, 205, 165), outline=WOOD_D, width=4)
    hx, hy, ang = q["head"]
    ang += extra_spin
    r = 62
    d.ellipse((hx - r, hy - r, hx + r, hy + r), fill=WOOD, outline=WOOD_D, width=6)

    def rot(x, y):
        c, s = math.cos(ang), math.sin(ang)
        return hx + x * c - y * s, hy + x * s + y * c

    for ex in (-20, 20):
        x, y = rot(ex, -8)
        d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=INK)
    if face == "smile":
        pts = [rot(18 * math.cos(t), 14 + 10 * math.sin(t)) for t in np.linspace(0.2, math.pi - 0.2, 12)]
        d.line(pts, fill=INK, width=5)
    else:  # 真顔
        d.line((rot(-14, 22), rot(14, 22)), fill=INK, width=5)


def draw_leaves(d, alpha):
    if alpha <= 0:
        return
    bg = (247, 243, 234)
    for outline in (LEAF_L, LEAF_R):
        d.polygon([tuple(p) for p in outline], fill=mix(bg, LEAF, alpha), outline=mix(bg, (70, 120, 170), alpha))


def sparkle(d, t):
    rng = np.random.default_rng(5)
    for k in range(14):
        x, y = rng.uniform(120, 960), rng.uniform(420, 1350)
        ph = (t * 2.2 + rng.uniform()) % 1
        s = 26 * math.sin(math.pi * ph)
        if s > 1:
            d.polygon([(x, y - s), (x + s * 0.25, y - s * 0.25), (x + s, y), (x + s * 0.25, y + s * 0.25),
                       (x, y + s), (x - s * 0.25, y + s * 0.25), (x - s, y), (x - s * 0.25, y - s * 0.25)],
                      fill=(250, 200, 40))


def floor(d):
    d.line((60, 1520, 1020, 1520), fill=(200, 188, 168), width=6)


# ------------------------------------------------------------------ 構成
def build(out_path, previews=()):
    xs = [250, 540, 830]
    BASE = 1150
    stand = [pose_stand(x, BASE) for x in xs]
    stretch = [pose_stretch(x, BASE) for x in xs]
    bent = [pose_bent(x, BASE, k=i * 0.7) for i, x in enumerate(xs)]
    leaf = [pose_leaf_outline(LEAF_L, -TILT), pose_leaf_outline(LEAF_R, TILT), pose_leaf_veins()]
    audio = Audio(40)
    scenes = []
    tc = [0.0]

    def add(dur, fn, sfx=None):
        if sfx:
            sfx(tc[0])
        scenes.append((dur, fn))
        tc[0] += dur

    def hop(i, t):
        return max(0.0, math.sin(t * 7 - i * 0.9)) * 30

    def shifted(q, dy):
        return {k: (v + [0, -dy] if k != "head" else (v[0], v[1] - dy, v[2])) for k, v in q.items()}

    def intro(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            w = max(0.0, math.sin(t * 9)) if i == 1 else 0.0
            draw_puppet(d, shifted(pose_stand(xs[i], BASE, wave=w), hop(i, t) * (t < 1.2)))
        text(d, (W / 2, 330), "東大の\nイチョウマーク", int(86 * (0.6 + 0.4 * ease_out_back(t / 0.4))), stroke=6)
        caption(d, "この3人で作ります", sub="（よろしくお願いします）", pop=t / 0.25)

    add(2.6, intro, lambda t: [audio.pop(t + k * 0.25, 600 + 120 * k) for k in range(3)])

    def to_stretch(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            draw_puppet(d, interp_pose(stand[i], stretch[i], ease_out_back(u * 1.3, 1.2), lag=0.1))
        caption(d, "まず腕を伸ばします", sub="（伸びます）" if u > 0.35 else None, pop=t / 0.25)

    add(2.2, to_stretch, lambda t: audio.sweep(t + 0.1, 180, 900, 0.8, 0.35, wobble=0.25))

    def to_bent(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            draw_puppet(d, interp_pose(stretch[i], bent[i], u, lag=0.3), face="plain" if u > 0.5 else "smile")
        caption(d, "関節を逆に曲げて", sub="（曲がります）" if u > 0.35 else None, pop=t / 0.25)
        if u > 0.5:
            text(d, (W / 2, 290), "ボキッ", 80, fill=(210, 30, 40), stroke=6)

    add(2.4, to_bent, lambda t: [audio.crack(t + 0.8 + k * 0.13) for k in range(6)])

    def to_leaf(img, d, t, u):
        header(d, TITLE)
        floor(d)
        draw_leaves(d, clamp01((u - 0.8) / 0.2) * 0.35)
        for i in range(3):
            draw_puppet(d, interp_pose(bent[i], leaf[i], u, lag=0.35), face="plain",
                        extra_spin=4 * math.pi * ease(u))
        caption(d, "巻きつけて…\n首は2回転します", sub="（回ります）", pop=t / 0.25)

    add(3.0, to_leaf, lambda t: audio.sweep(t, 300, 1200, 2.4, 0.18, wobble=0.4))

    def done(img, d, t, u):
        header(d, TITLE)
        floor(d)
        draw_leaves(d, 0.35 + 0.65 * ease(t / 0.6))
        for i in range(3):
            draw_puppet(d, leaf[i], face="smile")
        sparkle(d, t)
        caption(d, "完成！\n東大のイチョウ！", pop=t / 0.3)

    add(2.6, done, lambda t: audio.chime(t))

    def note(img, d, t, u):
        done(img, d, t + 2.6, 1)
        text(d, (W / 2, 300), "※関節は考慮していません", 54, fill=(210, 30, 40), stroke=6)

    add(1.6, note, lambda t: audio.pop(t, 330))

    def punch(img, d, t, u):
        header(d, TITLE)
        floor(d)
        draw_leaves(d, 1)
        for i in range(3):
            draw_puppet(d, leaf[i], face="smile")
        tsukkomi(img, t / 2.5)

    add(2.6, punch, lambda t: audio.don(t))

    total = render(scenes, out_path, audio, preview_times=previews)
    print(f"wrote {out_path} ({total:.1f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "video2_ginkgo.mp4"
    build(out, previews=[1.5, 3.8, 6.2, 7.5, 9.0, 11.5, 13.0, 15.5] if "--preview" in sys.argv else [])
