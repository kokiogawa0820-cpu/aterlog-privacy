"""動画2: ピクトグラム3人が, 伸び縮み・変形・曲がりだけでイチョウマークになる.
(体は分裂しない. ツッコミは撮影側で入れるので動画には入れない)

※ 東京大学の公式ロゴそのものではなく, 2枚のイチョウの葉を組み合わせたそれ風のシルエットです.
"""
import math
import sys

import numpy as np

from common import W, Audio, caption, clamp01, ease, ease_out_back, header, mix, render, text

TITLE = "3人でイチョウマーク作ります"
PICTO = (28, 56, 104)  # ピクトグラムの色
LEAF = (126, 178, 216)  # 淡青っぽい色
BG = (247, 243, 234)
NP = 25  # 各パーツのサンプル点数
LIMBS = ["torso", "armL", "armR", "legL", "legR"]
LIMB_W = 32
HEAD_R = 40
HEAD_GAP = 14


def chain(pts, n=NP):
    """関節 (3点) を通る折れ線を n 点にする."""
    a, b, c = (np.array(p, float) for p in pts)
    h = (n - 1) // 2
    s = np.linspace(0, 1, h + 1)[:, None]
    return np.vstack([a + (b - a) * s, (b + (c - b) * s)[1:]])


def resample(poly, n=NP):
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n)
    return np.column_stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])])


def connect(q):
    """腕は肩 (胴の2点目) に, 脚は腰 (胴の最後) にくっつける. 体が分裂しないための拘束."""
    q = {k: v.copy() for k, v in q.items()}
    w = ((1 - np.linspace(0, 1, NP)) ** 2)[:, None]
    for name, anchor in (("armL", q["torso"][1]), ("armR", q["torso"][1]),
                         ("legL", q["torso"][-1]), ("legR", q["torso"][-1])):
        q[name] += (anchor - q[name][0]) * w
    return q


def head_pos(q):
    """頭は首の先 (胴の向きの延長) に, 少しすき間をあけて付く."""
    neck, below = q["torso"][0], q["torso"][2]
    v = neck - below
    v /= np.linalg.norm(v) + 1e-9
    return neck + v * (HEAD_R + HEAD_GAP)


# ------------------------------------------------------------------ ポーズ
def pose_stand(cx, base, s=1.6, wave=0.0):
    p = lambda x, y: (cx + x * s, base + y * s)
    return {
        "torso": chain([p(0, -150), p(0, -75), p(0, 0)]),
        "armL": chain([p(0, -138), p(-44 - 20 * wave, -80 - 50 * wave), p(-54 - 10 * wave, -18 - 130 * wave)]),
        "armR": chain([p(0, -138), p(44, -80), p(54, -18)]),
        "legL": chain([p(0, 0), p(-22, 85), p(-30, 170)]),
        "legR": chain([p(0, 0), p(22, 85), p(30, 170)]),
    }


def pose_stretch(cx, base, s=1.6):
    """腕は3倍に伸びて真上, 胴は伸びて, 脚は縮む (足は床のまま)."""
    q = pose_stand(cx, base, s)
    neck = np.array([cx, base - 150 * s - 60])
    hip = np.array([cx, base + 170 * s - 90])
    foot_y = base + 170 * s
    q["torso"] = chain([neck, (neck + hip) / 2, hip])
    sh = q["torso"][1]
    for side, sx in (("armL", -1), ("armR", 1)):
        q[side] = chain([sh, sh + [sx * 40, -250], sh + [sx * 70, -520]])
    for side, sx in (("legL", -1), ("legR", 1)):
        q[side] = chain([hip, (hip[0] + sx * 28, (hip[1] + foot_y) / 2), (hip[0] + sx * 40, foot_y)])
    return q


def spiral_from(start, r0, r1, a0, turns, n=NP):
    """start から始まって内側へ巻くうず巻き."""
    c = np.array(start) - r0 * np.array([math.cos(a0), math.sin(a0)])
    a = a0 + np.linspace(0, 2 * math.pi * turns, n)
    r = np.linspace(r0, r1, n)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def pose_bent(cx, base, s=1.6, k=0.0):
    """関節が逆に曲がる: 胴はS字, 腕はうず巻き, 膝は逆くの字."""
    neck = np.array([cx + 30, base - 150 * s])
    y = np.linspace(neck[1], base, NP)
    torso = np.column_stack([cx + 55 * np.sin(np.linspace(0, 2 * math.pi, NP) + k) * np.linspace(0.3, 1, NP), y])
    torso[0] = neck
    sh = torso[1]
    hip = torso[-1]
    return {
        "torso": torso,
        "armL": spiral_from(sh, 95, 12, math.pi * 0.05 + k, -1.4),
        "armR": spiral_from(sh, 95, 12, math.pi * 0.95 - k, 1.4),
        "legL": chain([hip, hip + [-120, 70], hip + [-20, 272]]),
        "legR": chain([hip, hip + [120, 70], hip + [20, 272]]),
    }


# ------------------------------------------------------------------ イチョウ
BX, BY = 540, 1150  # 2枚の葉の付け根
RAD = 440
SPREAD = math.radians(30)
TILT = math.radians(36)


def polar(phi, r):
    return np.array([BX + r * math.sin(phi), BY - r * math.cos(phi)])


def leaf_outline(axis, n=201):
    """付け根 → 左の辺 → 上の縁 (中央に切れ込み) → 右の辺 → 付け根. 真ん中の点が切れ込み."""
    def top_r(phi):
        d = phi - axis
        notch = 0.27 * math.exp(-(d / 0.09) ** 2)
        return RAD * (1 - notch + 0.018 * math.sin(14 * d) - 0.10 * (d / SPREAD) ** 4)

    left = [polar(axis - SPREAD, RAD * f) for f in np.linspace(0, 1, 30) ** 1.2]
    top = [polar(p, top_r(p)) for p in np.linspace(axis - SPREAD, axis + SPREAD, 121)]
    right = [polar(axis + SPREAD, RAD * f) for f in np.linspace(1, 0, 30) ** 1.2]
    # 付け根が少しくびれるよう左右の辺を内側にしならせる
    left = [(x + (BX - x) * 0.12 * math.sin(math.pi * k / 29), y) for k, (x, y) in enumerate(left)]
    right = [(x + (BX - x) * 0.12 * math.sin(math.pi * k / 29), y) for k, (x, y) in enumerate(right)]
    return resample(left + top + right, n)


LEAF_L = leaf_outline(-TILT)
LEAF_R = leaf_outline(TILT)


def pose_leaf(outline, axis):
    """1枚の葉を1人で: 首は切れ込み, 両腕で葉の縁をぐるっと, 胴と両脚で中央の葉脈."""
    mid = len(outline) // 2
    neck = outline[mid] + (polar(axis, 0) - outline[mid]) * 0.02
    hip = polar(axis, RAD * 0.42)
    return {
        "torso": resample([neck, hip]),
        "armL": resample(outline[mid:8:-1]),
        "armR": resample(outline[mid:-9]),
        "legL": chain([hip, polar(axis - 0.03, RAD * 0.24), polar(axis - 0.06, RAD * 0.07)]),
        "legR": chain([hip, polar(axis + 0.03, RAD * 0.24), polar(axis + 0.06, RAD * 0.07)]),
    }


def pose_stem():
    """3人目は逆立ちで茎に: 頭が下, 両脚は付け根で2枚の葉を支える."""
    neck = np.array([BX + 6, BY + 250])
    hip = np.array([BX, BY + 10])
    return {
        "torso": resample([neck, (BX + 22, BY + 130), hip]),
        "armL": chain([neck + [0, -18], neck + [-70, -40], neck + [-105, -95]]),
        "armR": chain([neck + [0, -18], neck + [70, -30], neck + [100, -80]]),
        "legL": chain([hip, polar(-0.35, RAD * 0.07), polar(-0.5, RAD * 0.13)]),
        "legR": chain([hip, polar(0.35, RAD * 0.07), polar(0.5, RAD * 0.13)]),
    }


# ------------------------------------------------------------------ 描画
def interp_pose(a, b, u, lag=0.25):
    """パーツの先ほど遅れて動く → ぐにゃっとした変形."""
    out = {}
    for name in LIMBS:
        delay = np.linspace(0, lag, NP)[:, None]
        uu = np.clip((u - delay) / (1 - lag), 0, 1)
        uu = uu * uu * (3 - 2 * uu)
        out[name] = a[name] + (b[name] - a[name]) * uu
    return out


def draw_picto(d, q, color=PICTO):
    q = connect(q)
    r = LIMB_W / 2
    for name in ["legL", "legR", "armL", "armR", "torso"]:
        pts = [tuple(p) for p in q[name]]
        d.line(pts, fill=color, width=LIMB_W, joint="curve")
        for x, y in pts:  # 継ぎ目のすき間を埋める
            d.ellipse((x - r, y - r, x + r, y + r), fill=color)
    tr = LIMB_W * 0.62  # 胴は少し太く
    d.line([tuple(p) for p in q["torso"]], fill=color, width=int(tr * 2), joint="curve")
    for x, y in (q["torso"][0], q["torso"][-1]):
        d.ellipse((x - tr, y - tr, x + tr, y + tr), fill=color)
    hx, hy = head_pos(q)
    d.ellipse((hx - HEAD_R, hy - HEAD_R, hx + HEAD_R, hy + HEAD_R), fill=color)


def draw_leaves(d, alpha):
    if alpha <= 0:
        return
    for outline in (LEAF_L, LEAF_R):
        d.polygon([tuple(p) for p in outline], fill=mix(BG, LEAF, alpha))


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
    BASE = 1248  # 腰の高さ (足が床に付く)
    stand = [pose_stand(x, BASE) for x in xs]
    stretch = [pose_stretch(x, BASE) for x in xs]
    bent = [pose_bent(x, BASE, k=i * 0.9) for i, x in enumerate(xs)]
    leaf = [pose_leaf(LEAF_L, -TILT), pose_leaf(LEAF_R, TILT), pose_stem()]
    audio = Audio(40)
    scenes = []
    tc = [0.0]

    def add(dur, fn, sfx=None):
        if sfx:
            sfx(tc[0])
        scenes.append((dur, fn))
        tc[0] += dur

    def shifted(q, dy):
        return {k: v + [0, -dy] for k, v in q.items()}

    def intro(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            w = max(0.0, math.sin(t * 9)) if i == 1 else 0.0
            hop = max(0.0, math.sin(t * 7 - i * 0.9)) * 30 * (t < 1.2)
            draw_picto(d, shifted(pose_stand(xs[i], BASE, wave=w), hop))
        text(d, (W / 2, 360), "東大の\nイチョウマーク", int(86 * (0.6 + 0.4 * ease_out_back(t / 0.4))), stroke=6)
        caption(d, "この3人で作ります", sub="（よろしくお願いします）", pop=t / 0.25)

    add(2.4, intro, lambda t: [audio.pop(t + k * 0.25, 600 + 120 * k) for k in range(3)])

    def to_stretch(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            draw_picto(d, interp_pose(stand[i], stretch[i], ease_out_back(u * 1.3, 1.2), lag=0.1))
        caption(d, "腕と胴を伸ばして\n脚を縮めます", sub="（伸び縮みします）" if u > 0.35 else None, pop=t / 0.25)

    add(2.3, to_stretch, lambda t: [audio.sweep(t + 0.1, 180, 900, 0.7, 0.35, wobble=0.25),
                                    audio.sweep(t + 0.8, 700, 250, 0.4, 0.25)])

    def to_bent(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            draw_picto(d, interp_pose(stretch[i], bent[i], u, lag=0.3))
        caption(d, "関節を逆に曲げて", sub="（曲がります）" if u > 0.35 else None, pop=t / 0.25)
        if u > 0.5:
            text(d, (W / 2, 290), "ボキッ", 80, fill=(210, 30, 40), stroke=6)

    add(2.4, to_bent, lambda t: [audio.crack(t + 0.8 + k * 0.13) for k in range(6)])

    def to_leaf(img, d, t, u):
        header(d, TITLE)
        floor(d)
        draw_leaves(d, clamp01((u - 0.8) / 0.2) * 0.35)
        for i in range(3):
            draw_picto(d, interp_pose(bent[i], leaf[i], u, lag=0.35))
        caption(d, "しならせて\n巻きつけて…", sub="（1人は逆立ちで茎）" if u > 0.4 else None, pop=t / 0.25)

    add(3.0, to_leaf, lambda t: audio.sweep(t, 300, 1200, 2.4, 0.18, wobble=0.4))

    def done(img, d, t, u, note=False):
        header(d, TITLE)
        floor(d)
        draw_leaves(d, 0.35 + 0.65 * ease(t / 0.6))
        for i in range(3):
            draw_picto(d, leaf[i])
        sparkle(d, t)
        caption(d, "完成！\n東大のイチョウ！", pop=t / 0.3 if not note else 1)
        if note:
            text(d, (W / 2, 300), "※関節は考慮していません", 54, fill=(210, 30, 40), stroke=6)

    add(2.4, done, lambda t: audio.chime(t))
    # 最後はそのまま止めておく (ここで撮影側がツッコミを入れる)
    add(3.0, lambda img, d, t, u: done(img, d, t + 2.4, u, note=True), lambda t: audio.pop(t, 330))

    total = render(scenes, out_path, audio, preview_times=previews)
    print(f"wrote {out_path} ({total:.1f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "video2_ginkgo.mp4"
    build(out, previews=[1.2, 3.0, 4.6, 6.2, 8.0, 9.5, 11.0, 14.0] if "--preview" in sys.argv else [])
