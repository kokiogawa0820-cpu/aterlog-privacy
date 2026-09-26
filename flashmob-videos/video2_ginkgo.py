"""動画2: ピクトグラム3人が, 最初は普通に動き, だんだん伸び縮み・変形・曲がりで東大のイチョウマークになる.
(体は分裂しない. ツッコミは撮影側で入れるので動画には入れない)

最後に本物のマーク (assets/utokyo_mark.png, trace_mark.py で作成) を重ねて浮き上がらせ, 横に並べる.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common import BG, H, Audio, caption, clamp01, ease, ease_out_back, header, mix, render, text

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
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "utokyo_logo.png")
LOGO_W = 330  # 右側に出す東大マーク (文字付き) の幅
LOGO_ASPECT = 1430 / 1000  # utokyo_logo.png の 高さ/幅
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


def angles(cx, tilt=180, armL=(-37, -9), armR=(37, 9), legL=(-14.5, -5), legR=(14.5, 5),
           armk=1.0, bend=0.0, noodle=0.0):
    """関節角度のパラメータ. 腰の高さは足が床に付くよう自動で決める.
    armk: 腕の長さの倍率, bend: 胴のしなり (px), noodle: 腕のぐにゃり (px). 普通の人は 1, 0, 0."""
    return {"cx": cx, "tilt": tilt, "armL": armL, "armR": armR, "legL": legL, "legR": legR,
            "armk": armk, "bend": bend, "noodle": noodle}


def bow(pts, amount):
    """折れ線の両端はそのまま, 真ん中ほど横にふくらませる."""
    if abs(amount) < 1e-6:
        return pts
    v = pts[-1] - pts[0]
    n = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-9)
    return pts + n * (amount * np.sin(np.linspace(0, math.pi, len(pts))))[:, None]


def build_pose(p):
    drop = max(LEN["thigh"] * unit(p[k][0])[1] + LEN["shin"] * unit(p[k][1])[1] for k in ("legL", "legR"))
    hip = np.array([p["cx"], FLOOR_Y - drop * S])
    neck = hip + unit(p["tilt"]) * LEN["torso"] * S
    torso = bow(resample([neck, hip]), p["bend"])
    q = {"torso": torso}
    for k, sgn in (("armL", 1), ("armR", -1)):
        e = torso[1] + unit(p[k][0]) * LEN["upper"] * S * p["armk"]
        q[k] = bow(chain([torso[1], e, e + unit(p[k][1]) * LEN["lower"] * S * p["armk"]]), sgn * p["noodle"])
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


_logo = {}


def paste_logo(img, card, x0, y0, v):
    """白い台紙 + 東大マーク (文字付き) を, 透明度 v で貼る."""
    if "img" not in _logo:
        lg = Image.open(LOGO_PATH).convert("RGBA")
        _logo["img"] = lg.resize((LOGO_W, int(LOGO_W * lg.height / lg.width)), Image.LANCZOS)
    lg = _logo["img"]
    layer = card.copy()
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, card.size[0] - 1, card.size[1] - 1), radius=24, fill=int(235 * v))
    layer.putalpha(mask)
    shadow = Image.new("RGBA", (card.size[0] + 40, card.size[1] + 40), (0, 0, 0, 0))
    sm = Image.new("L", shadow.size, 0)
    ImageDraw.Draw(sm).rounded_rectangle((20, 20, card.size[0] + 20, card.size[1] + 20), radius=24, fill=int(60 * v))
    shadow.putalpha(sm.filter(ImageFilter.GaussianBlur(12)))
    x, y = int(x0 - 30), int(y0 - 25)
    img.paste(shadow, (x - 12, y - 6), shadow)
    img.paste(layer, (x, y), layer)
    logo = lg.copy()
    logo.putalpha(lg.getchannel("A").point(lambda a: int(a * v)))
    img.paste(logo, (int(x0), int(y0)), logo)


def floor(d):
    d.line((60, FLOOR_Y, 1020, FLOOR_Y), fill=(200, 188, 168), width=6)


# ------------------------------------------------------------------ 横長の画面への配置
# 人の動きは縦長 (1080x1920) の座標で作ってあるので, その一部を切り出して画面左側に縮小して置く.
# 右側はテロップと東大マーク用.
OW, OH = 1080, 1920
CROP = (0, 380, 1080, 1560)
DEST_H = H - 150
DEST_W = int(DEST_H * (CROP[2] - CROP[0]) / (CROP[3] - CROP[1]))
DEST_XY = (90, 130)
PANEL_X, PANEL_W = 1410, 820
_main = {}


def staged(fn):
    """fn(layer, layer_draw, t, u) は縦長の座標で人を描く. テロップ等は cap() / main() で右側に描く."""
    def wrapped(img, d, t, u):
        _main["img"], _main["d"] = img, d
        lay = Image.new("RGB", (OW, OH), BG)
        ld = ImageDraw.Draw(lay)
        floor(ld)
        fn(lay, ld, t, u)
        img.paste(lay.crop(CROP).resize((DEST_W, DEST_H), Image.BICUBIC), DEST_XY)
        header(d, TITLE)
    return wrapped


def main():
    return _main["img"], _main["d"]


def cap(s, y=850, **kw):
    caption(_main["d"], s, x=PANEL_X, y=y, width=PANEL_W, **kw)


# ------------------------------------------------------------------ 構成
def add_long_part(add, audio, xs, poses):
    """長い版だけの追加パート: 普通の動き2つ → すこーーーしだけ変な動き2つ. 最後のポーズを返す."""
    one_leg = [angles(x, armL=(-150, 195), armR=(150, -195), legL=(-5, -2), legR=(75, 5)) for x in xs]
    side = [angles(x, tilt=205, armL=(-135, 200), armR=(170, -160), legL=(-22, -12), legR=(22, 12)) for x in xs]
    vpose = dict(armL=(-150, 200), armR=(150, -200), legL=(-20, -10), legR=(20, 10))
    longer = [angles(x, armk=1.35, **vpose) for x in xs]

    def move(src, dst, label, jiggle=0.0):
        def fn(img, d, t, u):
            for i in range(3):
                p = lerp_angles(src[i], dst[i], u * 1.5)
                if jiggle:  # 胴と腕がちょっとだけぐにゃぐにゃ (最後は元に戻る)
                    w = math.sin(2 * math.pi * 2 * u) * math.sin(math.pi * u)
                    p["bend"] = jiggle * w
                    p["noodle"] = jiggle * 0.9 * w
                draw_picto(d, build_pose(p))
            cap(label, pop=t / 0.25)
        return fn

    add(1.9, move(poses, one_leg, "そろって片足立ち"), lambda t: audio.pop(t + 0.15, 620, 0.3))
    add(1.9, move(one_leg, side, "体を横に倒して"), lambda t: audio.whoosh(t + 0.1, 0.5, 0.2))
    # ここから, すこーーーしだけ変
    add(2.0, move(side, longer, "腕を少し伸ばして"), lambda t: audio.sweep(t + 0.3, 300, 420, 0.5, 0.15))
    add(2.2, move(longer, longer, "体をやわらかく", jiggle=45),
        lambda t: audio.sweep(t + 0.1, 350, 300, 1.6, 0.12, wobble=0.3))
    return longer



def build(out_path, previews=(), long=False):
    """long=True: 普通の動きを増やし, 「ちょっとだけ変」を経てから伸びるパートへ行く版."""
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
        scenes.append((dur, staged(fn)))
        tc[0] += dur

    def shifted(q, dy):
        return {k: v + [0, -dy] for k, v in q.items()}

    def intro(img, d, t, u):
        for i in range(3):
            p = dict(stand[i])
            if i == 1:  # 真ん中の人は手を振る
                w = math.sin(t * 9)
                p["armR"] = (150 + 12 * w, 170 + 25 * w)
            hop = max(0.0, math.sin(t * 7 - i * 0.9)) * 30 * (t < 1.2)
            draw_picto(d, shifted(build_pose(p), hop))
        text(main()[1], (PANEL_X, 420), "東大の\nイチョウマーク", int(100 * (0.6 + 0.4 * ease_out_back(t / 0.4))),
             stroke=6)
        cap("この3人で作ります", pop=t / 0.25)

    add(2.2, intro, lambda t: [audio.pop(t + k * 0.25, 600 + 120 * k) for k in range(3)])

    # --- ここまでは普通の動き (関節角度で動かすので無理がない)
    def natural1(img, d, t, u):
        for i in range(3):  # 肘を曲げながら上げる (途中で手が隣の人に当たらないように)
            v = clamp01(u * 1.5)
            p = lerp_angles(stand[i], raising[i], v * 2) if v < 0.5 else lerp_angles(raising[i], tpose[i], v * 2 - 1)
            draw_picto(d, build_pose(p))
        cap("まずは両手を上げて", pop=t / 0.25)

    add(1.9, natural1, lambda t: audio.whoosh(t + 0.1, 0.5, 0.2))

    def natural2(img, d, t, u):
        for i in range(3):
            draw_picto(d, build_pose(lerp_angles(tpose[i], poses[i], u * 1.5)))
        cap("それぞれポーズ", pop=t / 0.25)

    add(2.1, natural2, lambda t: [audio.pop(t + 0.1 + k * 0.15, 700 + 90 * k, 0.3) for k in range(3)])

    last = poses
    if long:
        last = add_long_part(add, audio, xs, poses)

    # --- ここから無理な動き
    posed = [build_pose(p) for p in last]

    def to_stretch(img, d, t, u):
        for i in range(3):
            draw_picto(d, interp_pose(posed[i], stretch[i], ease_out_back(u * 1.3, 1.2), lag=0.1))
        cap("腕と胴を伸ばして\n脚を縮めます", pop=t / 0.25)

    add(2.2, to_stretch, lambda t: [audio.sweep(t + 0.1, 180, 900, 0.7, 0.35, wobble=0.25),
                                    audio.sweep(t + 0.8, 700, 250, 0.4, 0.25)])

    def to_bent(img, d, t, u):
        for i in range(3):
            draw_picto(d, interp_pose(stretch[i], bent[i], u, lag=0.3))
        cap("関節を逆に曲げて", pop=t / 0.25)

    add(2.3, to_bent, lambda t: audio.sweep(t + 0.3, 500, 250, 1.2, 0.2, wobble=0.3))

    def to_mark(img, d, t, u):
        for i in draw_order:
            draw_picto(d, interp_pose(bent[i], mark[i], u, lag=0.35))
        cap("しならせて\n巻きつけて…", pop=t / 0.25)

    add(3.0, to_mark, lambda t: audio.sweep(t, 300, 1200, 2.4, 0.18, wobble=0.4))

    def colored(img, d, v):
        """色が付いてマークになる. v: 0→1."""
        paste_mark(img, 540, 930, MARK_SIZE, alpha=0.55 * v)
        d = ImageDraw.Draw(img)
        for i in draw_order:
            draw_picto(d, mark[i], color=mix(PICTO, final_color[i], v))
        return d

    def formed(img, d, t, u):
        """形が完成したところで一瞬止める (まだ色は付かない)."""
        colored(img, d, 0)
        cap("しならせて\n巻きつけて…")

    add(0.8, formed)

    color_dur = 3.2

    def logo_in(v):
        """右側に東大マーク (文字付き) が浮き出てくる. v: 0→1."""
        if v <= 0:
            return
        img = main()[0]
        x0, y0 = PANEL_X - LOGO_W / 2, 170 + 40 * (1 - v)
        h = LOGO_W * LOGO_ASPECT
        card = Image.new("RGBA", (int(LOGO_W + 60), int(h + 50)), (255, 255, 255, int(235 * v)))
        paste_logo(img, card, x0, y0, v)

    def coloring(img, d, t, u):
        v = ease(t / (color_dur - 0.3))
        d = colored(img, d, v)
        logo_in(v)
        if t > 1.0:
            cap("完成！\n東大のイチョウマーク！", pop=(t - 1.0) / 0.3)
        else:
            cap("しならせて\n巻きつけて…")

    add(color_dur, coloring, lambda t: [audio.sweep(t, 400, 800, 2.5, 0.08), audio.chime(t + 1.0)])

    def hold(img, d, t, u):
        d = colored(img, d, 1)
        logo_in(1)
        cap("完成！\n東大のイチョウマーク！")

    # 最後はそのまま止めておく (ここで撮影側がツッコミを入れる)
    add(3.0, hold)

    total = render(scenes, out_path, audio, preview_times=previews)
    print(f"wrote {out_path} ({total:.1f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    long = "--long" in sys.argv
    out = args[0] if args else ("video2_ginkgo_long.mp4" if long else "video2_ginkgo.mp4")
    previews = [6.8, 7.5, 8.6, 9.5, 10.6, 11.6, 12.6, 13.4, 14.2, 15.0, 16.5, 27.0] if long else \
        [8.0, 12.0, 14.5, 15.3, 16.2, 17.2, 18.2, 20.5]
    build(out, previews=previews if "--preview" in sys.argv else [], long=long)
