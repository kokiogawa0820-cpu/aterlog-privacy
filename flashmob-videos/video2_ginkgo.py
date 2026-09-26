"""動画2: ピクトグラム3人が, 最初は普通に動き, だんだん伸び縮み・変形・曲がりで東大のイチョウマークになる.
(体は分裂しない. ツッコミは撮影側で入れるので動画には入れない)

最後に本物のマーク (assets/utokyo_mark.png, trace_mark.py で作成) を重ねて浮き上がらせ, 横に並べる.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import INK, W, Audio, caption, clamp01, ease, ease_out_back, header, mix, render, text

TITLE = "3人でイチョウマーク作ります"
PICTO = (28, 56, 104)  # ピクトグラムの色
YELLOW = (250, 189, 3)
BLUE = (24, 126, 196)
NP = 25  # 各パーツのサンプル点数
LIMBS = ["torso", "armL", "armR", "legL", "legR"]
LIMB_W = 32
HEAD_R = 40
HEAD_GAP = 14
FLOOR_Y = 1520
S = 1.6  # 人の大きさ
LEN = {"torso": 150, "upper": 73, "lower": 63, "thigh": 88, "shin": 85}

MARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "utokyo_mark.png")
MARK_SIZE = 780  # 画面上のマークの直径
MARK_X0, MARK_Y0 = 540 - MARK_SIZE / 2, 930 - MARK_SIZE / 2


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


# ------------------------------------------------------------------ 普通のポーズ (関節角度で作る)
def unit(deg):
    """0°=真下, 90°=右, 180°=真上, -90°=左."""
    r = math.radians(deg)
    return np.array([math.sin(r), math.cos(r)])


def angles(cx, tilt=180, armL=(-37, -9), armR=(37, 9), legL=(-14.5, -5), legR=(14.5, 5)):
    """関節角度のパラメータ. 腰の高さは足が床に付くよう自動で決める."""
    return {"cx": cx, "tilt": tilt, "armL": armL, "armR": armR, "legL": legL, "legR": legR}


def build_pose(p):
    drop = max(LEN["thigh"] * unit(p[k][0])[1] + LEN["shin"] * unit(p[k][1])[1] for k in ("legL", "legR"))
    hip = np.array([p["cx"], FLOOR_Y - drop * S])
    neck = hip + unit(p["tilt"]) * LEN["torso"] * S
    torso = resample([neck, hip])
    q = {"torso": torso}
    for k in ("armL", "armR"):
        e = torso[1] + unit(p[k][0]) * LEN["upper"] * S
        q[k] = chain([torso[1], e, e + unit(p[k][1]) * LEN["lower"] * S])
    for k in ("legL", "legR"):
        e = hip + unit(p[k][0]) * LEN["thigh"] * S
        q[k] = chain([hip, e, e + unit(p[k][1]) * LEN["shin"] * S])
    return q


def lerp_angles(a, b, u):
    u = ease(u)
    out = {}
    for k in a:
        va, vb = np.array(a[k], float), np.array(b[k], float)
        out[k] = va + (vb - va) * u
    return out


# ------------------------------------------------------------------ 無理なポーズ
def pose_stretch(cx):
    """腕は3倍に伸びて真上, 胴は伸びて, 脚は縮む (足は床のまま)."""
    base = FLOOR_Y - 272
    neck = np.array([cx, base - 150 * S - 60])
    hip = np.array([cx, FLOOR_Y - 90])
    q = {"torso": chain([neck, (neck + hip) / 2, hip])}
    sh = q["torso"][1]
    for side, sx in (("armL", -1), ("armR", 1)):
        q[side] = chain([sh, sh + [sx * 40, -250], sh + [sx * 70, -520]])
    for side, sx in (("legL", -1), ("legR", 1)):
        q[side] = chain([hip, (hip[0] + sx * 28, (hip[1] + FLOOR_Y) / 2), (hip[0] + sx * 40, FLOOR_Y)])
    return q


def spiral_from(start, r0, r1, a0, turns, n=NP):
    """start から始まって内側へ巻くうず巻き."""
    c = np.array(start) - r0 * np.array([math.cos(a0), math.sin(a0)])
    a = a0 + np.linspace(0, 2 * math.pi * turns, n)
    r = np.linspace(r0, r1, n)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def pose_bent(cx, k=0.0):
    """関節が逆に曲がる: 胴はS字, 腕はうず巻き, 膝は逆くの字."""
    base = FLOOR_Y - 272
    neck = np.array([cx + 30, base - 150 * S])
    y = np.linspace(neck[1], base, NP)
    torso = np.column_stack([cx + 55 * np.sin(np.linspace(0, 2 * math.pi, NP) + k) * np.linspace(0.3, 1, NP), y])
    torso[0] = neck
    sh, hip = torso[1], torso[-1]
    return {
        "torso": torso,
        "armL": spiral_from(sh, 95, 12, math.pi * 0.05 + k, -1.4),
        "armR": spiral_from(sh, 95, 12, math.pi * 0.95 - k, 1.4),
        "legL": chain([hip, hip + [-120, 70], hip + [-20, 272]]),
        "legR": chain([hip, hip + [120, 70], hip + [20, 272]]),
    }


# ------------------------------------------------------------------ イチョウマーク (本物に合わせた形)
def M(*pts):
    """マーク画像 (1000x1000) 上の座標 → 画面座標."""
    k = MARK_SIZE / 1000
    return [(MARK_X0 + x * k, MARK_Y0 + y * k) for x, y in pts]


RING_C, RING_R = (300, 712), 160  # 青い輪 (マーク座標)


def ring(a0, a1, n=40):
    return M(*[(RING_C[0] + RING_R * math.cos(math.radians(a)), RING_C[1] + RING_R * math.sin(math.radians(a)))
               for a in np.linspace(a0, a1, n)])


def pose_yellow():
    """黄色の葉: 首は真ん中の切れ込み, 両腕で4つの葉先をなぞり, 左脚がくるっと回る茎."""
    return {
        "torso": resample(M((475, 330), (470, 600))),
        "armL": resample(M((470, 320), (458, 150), (462, 30), (380, 38), (250, 95), (140, 185),
                           (290, 345), (95, 248), (45, 330), (18, 460))),
        "armR": resample(M((480, 320), (478, 150), (472, 25), (620, 40), (745, 88),
                           (660, 285), (815, 140), (890, 220), (952, 330))),
        "legL": resample(M((470, 600), (468, 700), (440, 820), (360, 895), (230, 885), (110, 795),
                           (50, 660), (35, 562))),
        "legR": resample(M((470, 600), (560, 585), (700, 450), (948, 345))),
    }


def pose_blue():
    """青い葉: 胴は葉の中心線, 腕で葉のふちと切れ込みをなぞる."""
    return {
        "torso": resample(M((800, 640), (560, 900))),
        "armL": resample(M((790, 650), (720, 562), (655, 525), (590, 640), (530, 780), (505, 905))),
        "armR": resample(M((805, 648), (880, 512), (982, 490), (975, 600), (930, 730), (880, 812),
                           (742, 742), (820, 862), (700, 935), (560, 975))),
        "legL": resample(M((560, 900), (522, 940), (498, 972))),
        "legR": resample(M((560, 900), (600, 945), (640, 962))),
    }


def pose_ring():
    """青い輪: 胴は下の細い部分, 脚2本で輪をぐるっと."""
    hip = M((228, 900))[0]
    return {
        "torso": resample(M((432, 945), (330, 932), (228, 900))),
        "armL": resample(M((425, 946), (455, 958), (470, 962))),
        "armR": resample(M((425, 944), (450, 925), (462, 915))),
        "legL": resample([hip] + ring(125, 300)),
        "legR": resample([hip] + ring(125, 215)),
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


_mark_cache = {}


def mark_image(size):
    size = int(size)
    if size not in _mark_cache:
        _mark_cache[size] = Image.open(MARK_PATH).convert("RGBA").resize((size, size), Image.LANCZOS)
    return _mark_cache[size]


def paste_mark(img, cx, cy, size, alpha=1.0, shadow=0.0):
    m = mark_image(size)
    if alpha < 1:
        m = m.copy()
        m.putalpha(m.getchannel("A").point(lambda v: int(v * alpha)))
    x, y = int(cx - size / 2), int(cy - size / 2)
    if shadow > 0:
        sh = Image.new("RGBA", m.size, (0, 0, 0, 0))
        sh.putalpha(m.getchannel("A").point(lambda v: int(v * 0.35 * shadow)))
        sh = sh.filter(ImageFilter.GaussianBlur(14))
        img.paste(sh, (x + 12, y + 22), sh)
    img.paste(m, (x, y), m)


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
    d.line((60, FLOOR_Y, 1020, FLOOR_Y), fill=(200, 188, 168), width=6)


# ------------------------------------------------------------------ 構成
def build(out_path, previews=()):
    xs = [230, 540, 850]
    stand = [angles(x) for x in xs]
    tpose = [angles(x, armL=(-140, 210), armR=(140, -210), legL=(-20, -10), legR=(20, 10)) for x in xs]
    # 前腕は内側 (胸の前) を通って上げ下げする角度にしてある (途中で隣の人に当たらないように)
    raising = [angles(x, armL=(-80, 185), armR=(80, -185), legL=(-17, -7), legR=(17, 7)) for x in xs]
    poses = [  # それぞれ普通のポーズ
        angles(xs[0], armL=(-155, 190), armR=(45, -45), legL=(-18, -8), legR=(18, 8)),  # 片手を上, 片手は腰
        angles(xs[1], tilt=190, armL=(-120, 210), armR=(125, -210), legL=(-38, -38), legR=(5, 0)),  # 踏み込み
        angles(xs[2], armL=(-40, -15), armR=(40, 15), legL=(-62, 18), legR=(62, -18)),  # しゃがむ
    ]
    stretch = [pose_stretch(x) for x in xs]
    bent = [pose_bent(x, k=i * 0.9) for i, x in enumerate(xs)]
    mark = [pose_yellow(), pose_blue(), pose_ring()]
    final_color = [YELLOW, BLUE, BLUE]
    draw_order = [2, 1, 0]  # 輪 → 青い葉 → 黄色い葉 (茎が輪の上に来る)
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
            p = dict(stand[i])
            if i == 1:  # 真ん中の人は手を振る
                w = math.sin(t * 9)
                p["armR"] = (150 + 12 * w, 170 + 25 * w)
            hop = max(0.0, math.sin(t * 7 - i * 0.9)) * 30 * (t < 1.2)
            draw_picto(d, shifted(build_pose(p), hop))
        text(d, (W / 2, 360), "東大の\nイチョウマーク", int(86 * (0.6 + 0.4 * ease_out_back(t / 0.4))), stroke=6)
        caption(d, "この3人で作ります", sub="（よろしくお願いします）", pop=t / 0.25)

    add(2.2, intro, lambda t: [audio.pop(t + k * 0.25, 600 + 120 * k) for k in range(3)])

    # --- ここまでは普通の動き (関節角度で動かすので無理がない)
    def natural1(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):  # 肘を曲げながら上げる (途中で手が隣の人に当たらないように)
            v = clamp01(u * 1.5)
            p = lerp_angles(stand[i], raising[i], v * 2) if v < 0.5 else lerp_angles(raising[i], tpose[i], v * 2 - 1)
            draw_picto(d, build_pose(p))
        caption(d, "まずは両手を上げて", pop=t / 0.25)

    add(1.9, natural1, lambda t: audio.whoosh(t + 0.1, 0.5, 0.2))

    def natural2(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            draw_picto(d, build_pose(lerp_angles(tpose[i], poses[i], u * 1.5)))
        caption(d, "それぞれポーズ", sub="（ここまでは普通）" if u > 0.4 else None, pop=t / 0.25)

    add(2.1, natural2, lambda t: [audio.pop(t + 0.1 + k * 0.15, 700 + 90 * k, 0.3) for k in range(3)])

    # --- ここから無理な動き
    posed = [build_pose(p) for p in poses]

    def to_stretch(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            draw_picto(d, interp_pose(posed[i], stretch[i], ease_out_back(u * 1.3, 1.2), lag=0.1))
        caption(d, "腕と胴を伸ばして\n脚を縮めます", sub="（伸び縮みします）" if u > 0.35 else None, pop=t / 0.25)

    add(2.2, to_stretch, lambda t: [audio.sweep(t + 0.1, 180, 900, 0.7, 0.35, wobble=0.25),
                                    audio.sweep(t + 0.8, 700, 250, 0.4, 0.25)])

    def to_bent(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in range(3):
            draw_picto(d, interp_pose(stretch[i], bent[i], u, lag=0.3))
        caption(d, "関節を逆に曲げて", sub="（曲がります）" if u > 0.35 else None, pop=t / 0.25)
        if u > 0.5:
            text(d, (W / 2, 290), "ボキッ", 80, fill=(210, 30, 40), stroke=6)

    add(2.3, to_bent, lambda t: [audio.crack(t + 0.8 + k * 0.13) for k in range(6)])

    def to_mark(img, d, t, u):
        header(d, TITLE)
        floor(d)
        for i in draw_order:
            draw_picto(d, interp_pose(bent[i], mark[i], u, lag=0.35))
        caption(d, "しならせて\n巻きつけて…", sub="（1人は丸くなります）" if u > 0.4 else None, pop=t / 0.25)

    add(3.0, to_mark, lambda t: audio.sweep(t, 300, 1200, 2.4, 0.18, wobble=0.4))

    def colored(img, d, v):
        """色が付いてマークっぽくなる. v: 0→1."""
        paste_mark(img, 540, 930, MARK_SIZE, alpha=0.55 * v)
        d = ImageDraw.Draw(img)
        for i in draw_order:
            draw_picto(d, mark[i], color=mix(PICTO, final_color[i], v))
        return d

    def done(img, d, t, u):
        header(d, TITLE)
        floor(d)
        d = colored(img, d, ease(t / 0.8))
        sparkle(d, t)
        caption(d, "完成！", size=80, pop=t / 0.3)

    add(1.6, done, lambda t: audio.chime(t))

    card = (870, 360, 300)  # 本物カードの中心 x, y と大きさ

    def reveal(img, d, t, u):
        """本物のマークが重なって浮き上がる → 右上に移動して横に並ぶ."""
        header(d, TITLE)
        floor(d)
        d = colored(img, d, 1)
        if t < 1.3:  # 浮き上がる
            a = ease(t / 0.4)
            lift = ease(t / 0.6)
            paste_mark(img, 540, 930 - 20 * lift, MARK_SIZE * (1 + 0.04 * lift), alpha=0.95 * a, shadow=lift)
            if t > 0.4:
                text(d, (540, 930), "本物", 120, fill=(255, 255, 255), stroke=10, stroke_fill=INK)
        else:  # 右上へ移動
            k = ease((t - 1.3) / 0.6)
            cx = 540 + (card[0] - 540) * k
            cy = 910 + (card[1] - 910) * k
            size = MARK_SIZE * 1.04 + (card[2] - MARK_SIZE * 1.04) * k
            if k >= 1:
                draw_card(img, d)
            else:
                paste_mark(img, cx, cy, size, alpha=0.95, shadow=1 - k)
        caption(d, "東大のイチョウマーク！", pop=t / 0.3)

    def draw_card(img, d):
        cx, cy, size = card
        h = size / 2 + 20
        d.rounded_rectangle((cx - h, cy - h, cx + h, cy + h + 60), radius=26, fill=(255, 255, 255),
                            outline=INK, width=5)
        paste_mark(img, cx, cy, size)
        text(d, (cx, cy + h + 25), "本物", 44)

    add(2.4, reveal, lambda t: [audio.don(t + 0.05, 0.4), audio.chime(t + 1.4, 0.2)])

    def hold(img, d, t, u):
        header(d, TITLE)
        floor(d)
        d = colored(img, d, 1)
        draw_card(img, d)
        text(d, (820, 1370), "▲ 3人", 48)
        text(d, (330, 330), "※関節は\n考慮していません", 50, fill=(210, 30, 40), stroke=6)
        caption(d, "東大のイチョウマーク！")

    # 最後はそのまま止めておく (ここで撮影側がツッコミを入れる)
    add(3.0, hold, lambda t: audio.pop(t, 330))

    total = render(scenes, out_path, audio, preview_times=previews)
    print(f"wrote {out_path} ({total:.1f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "video2_ginkgo.mp4"
    build(out, previews=[1.0, 3.0, 5.5, 7.5, 9.0, 11.5, 13.0, 15.0, 15.8, 17.0, 18.5, 20.0]
          if "--preview" in sys.argv else [])
