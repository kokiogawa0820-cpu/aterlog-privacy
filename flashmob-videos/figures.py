"""人型のテイスト (描き方) いろいろ. 骨格 (胴・腕・脚の折れ線) は共通で, 描き方だけを差し替える.

骨格 q: {"torso": 首→腰, "armL"/"armR": 肩→手, "legL"/"legR": 腰→足} 各 25 点 (真ん中の 12 番が肘・膝).
最後に色が付く場面では tint (黄色/青) に amt (0→1) の割合で全体の色を寄せる.

STYLES:
  picto      ピクトグラム (今までのもの. video2_ginkgo.py 側で描く)
  mannequin  デッサン人形 (木製, 関節の球, 顔のない卵形の頭)
  plush      ぬいぐるみ (太く丸い手足, くまの頭, 縫い目)
  silhouette 人のシルエット (実際の体型に近い太さ, 手と靴)
  real       リアル寄りのイラスト (肌・髪・顔・パーティーの服装)
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from common import mix

STYLES = ["picto", "mannequin", "plush", "silhouette", "real"]
STYLE_NAMES = {"picto": "ピクトグラム", "mannequin": "デッサン人形", "plush": "ぬいぐるみ",
               "silhouette": "シルエット", "real": "リアル寄り"}
NP = 25
J = 12  # 肘・膝の番号


class Paint:
    """色に「最後の色付け」をかける."""

    def __init__(self, tint=None, amt=0.0):
        self.tint, self.amt = tint, amt

    def __call__(self, c, dark=0.0):
        if self.tint is None or self.amt <= 0:
            return c
        target = mix(self.tint, (0, 0, 0), dark) if dark else self.tint
        return mix(c, target, self.amt)


def normals(pts):
    t = np.gradient(pts, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.where(n > 1e-6, t / np.maximum(n, 1e-6), np.array([[0.0, 1.0]]))
    return np.column_stack([-t[:, 1], t[:, 0]])


def ribbon(d, pts, hw, fill, outline=None, ow=0, caps=True):
    """太さが変わる帯 (腕や脚). hw: 各点の半幅."""
    pts = np.asarray(pts, float)
    hw = np.broadcast_to(np.asarray(hw, float), (len(pts),))
    nv = normals(pts)
    for col, extra in ((outline, ow), (fill, 0)):
        if col is None:
            continue
        w = (hw + extra)[:, None]
        poly = np.vstack([pts + nv * w, (pts - nv * w)[::-1]])
        d.polygon([tuple(p) for p in poly], fill=col)
        if caps:
            for p, r in ((pts[0], w[0, 0]), (pts[-1], w[-1, 0])):
                d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=col)


def profile(knots):
    """[(番号, 半幅), ...] を 25 点ぶんの半幅にする."""
    k = np.array(knots, float)
    return np.interp(np.arange(NP), k[:, 0], k[:, 1])


def dot(d, p, r, fill, outline=None, ow=0):
    if outline is not None:
        d.ellipse((p[0] - r - ow, p[1] - r - ow, p[0] + r + ow, p[1] + r + ow), fill=outline)
    d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=fill)


def oval(d, center, direction, length, width, fill, outline=None, ow=0):
    """direction 向きに長い楕円 (手・足)."""
    u = direction / (np.linalg.norm(direction) + 1e-9)
    nv = np.array([-u[1], u[0]])
    a = np.linspace(0, 2 * math.pi, 24)
    for col, e in ((outline, ow), (fill, 0)):
        if col is None:
            continue
        pts = [tuple(center + u * (length / 2 + e) * math.cos(t) + nv * (width / 2 + e) * math.sin(t)) for t in a]
        d.polygon(pts, fill=col)


def up_vector(q):
    v = q["torso"][0] - q["torso"][2]
    return v / (np.linalg.norm(v) + 1e-9)


def paste_head(img, center, up, draw_fn, size=220):
    """頭は小さい画像に描いて, 首の向きに合わせて回して貼る."""
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(tile), size / 2)
    ang = math.degrees(math.atan2(up[0], -up[1]))
    tile = tile.rotate(-ang, resample=Image.BICUBIC)
    img.paste(tile, (int(center[0] - size / 2), int(center[1] - size / 2)), tile)


def end_dir(pts):
    v = pts[-1] - pts[-3]
    return v if np.linalg.norm(v) > 1e-6 else np.array([0.0, 1.0])


# ------------------------------------------------------------------ デッサン人形
WOOD, WOOD_D, WOOD_L = (226, 190, 142), (150, 108, 64), (238, 210, 170)


def draw_mannequin(img, d, q, who, P):
    w, wd, wl = P(WOOD), P(WOOD_D, 0.3), P(WOOD_L)
    arm = profile([(0, 11), (J - 1, 9), (J + 1, 9), (24, 7)])
    leg = profile([(0, 17), (J - 1, 12), (J + 1, 12), (24, 9)])
    for k in ("legL", "legR"):
        ribbon(d, q[k][:J + 1], leg[:J + 1], w, wd, 3)
        ribbon(d, q[k][J:], leg[J:], w, wd, 3)
        oval(d, q[k][-1] + end_dir(q[k]) * 0.12, end_dir(q[k]), 34, 16, w, wd, 3)
        dot(d, q[k][J], 12, wl, wd, 3)
    for k in ("armL", "armR"):
        ribbon(d, q[k][:J + 1], arm[:J + 1], w, wd, 3)
        ribbon(d, q[k][J:], arm[J:], w, wd, 3)
        oval(d, q[k][-1] + end_dir(q[k]) * 0.15, end_dir(q[k]), 26, 15, w, wd, 3)
        dot(d, q[k][J], 10, wl, wd, 3)
        dot(d, q[k][0], 12, wl, wd, 3)
    t = q["torso"]
    ribbon(d, t[:14], profile([(0, 14), (2, 36), (13, 22), (24, 22)])[:14], w, wd, 3)  # 胸
    ribbon(d, t[13:], profile([(0, 22), (11, 30)])[:12], w, wd, 3)  # 腰
    dot(d, t[13], 15, wl, wd, 3)
    dot(d, t[-1], 14, wl, wd, 3)
    for k in ("legL", "legR"):
        dot(d, q[k][0], 12, wl, wd, 3)
    up = up_vector(q)
    neck_top = t[0] + up * 26
    ribbon(d, np.array([t[0], neck_top]), 8, w, wd, 3)

    def head(hd, c):
        hd.ellipse((c - 40, c - 50, c + 40, c + 46), fill=wd)
        hd.ellipse((c - 37, c - 47, c + 37, c + 43), fill=w)
        hd.ellipse((c - 22, c - 38, c + 4, c - 18), fill=wl)  # つや
    paste_head(img, t[0] + up * 70, up, head)
    return 150  # 首から名札までの距離


# ------------------------------------------------------------------ ぬいぐるみ
PLUSH = [(244, 172, 182), (160, 212, 192), (192, 176, 226)]


def draw_plush(img, d, q, who, P):
    base = PLUSH[who % 3]
    c, cd, cl = P(base), P(mix(base, (60, 40, 50), 0.45), 0.3), P(mix(base, (255, 255, 255), 0.45))
    stitch = P((255, 255, 255))
    for k in ("legL", "legR", "armL", "armR"):
        hw = 25 if k.startswith("leg") else 21
        ribbon(d, q[k], hw, c, cd, 4)
        pts = q[k]
        for i in range(2, NP - 2, 2):  # 縫い目
            p = pts[i]
            d.ellipse((p[0] - 2.5, p[1] - 2.5, p[0] + 2.5, p[1] + 2.5), fill=stitch)
        dot(d, pts[-1], hw + 2, cl, cd, 4)  # 手先・足先は色違いの丸
    t = q["torso"]
    ribbon(d, t, profile([(0, 36), (8, 48), (24, 44)]), c, cd, 4)
    up = up_vector(q)
    tc = t[NP // 2]
    nv = np.array([-up[1], up[0]])
    belly = [tuple(tc + nv * 26 * math.cos(a) + up * 34 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 24)]
    d.polygon(belly, fill=cl)

    def head(hd, m):
        for ex in (-42, 42):  # 耳
            hd.ellipse((m + ex - 22, m - 66, m + ex + 22, m - 22), fill=cd)
            hd.ellipse((m + ex - 18, m - 62, m + ex + 18, m - 26), fill=c)
            hd.ellipse((m + ex - 9, m - 52, m + ex + 9, m - 36), fill=cl)
        hd.ellipse((m - 62, m - 56, m + 62, m + 56), fill=cd)
        hd.ellipse((m - 58, m - 52, m + 58, m + 52), fill=c)
        hd.ellipse((m - 26, m + 2, m + 26, m + 36), fill=cl)  # 口元
        hd.ellipse((m - 8, m + 6, m + 8, m + 18), fill=P((70, 45, 45)))  # 鼻
        for ex in (-24, 24):
            hd.ellipse((m + ex - 9, m - 18, m + ex + 9, m), fill=P((40, 30, 30)))  # ボタンの目
            hd.ellipse((m + ex - 3, m - 15, m + ex + 2, m - 10), fill=P((255, 255, 255)))
    paste_head(img, t[0] + up * 50, up, head)
    return 160


# ------------------------------------------------------------------ 体型 (シルエットとリアル寄りで共通)
ARM = profile([(0, 13), (J, 10), (24, 7)])
LEG = profile([(0, 21), (J, 14), (16, 15), (24, 8)])
TORSO = profile([(0, 11), (2, 40), (14, 27), (24, 33)])


def skirt_poly(q, flare=10):
    """ワンピースのスカート: 腰から左右の太ももの外側へ広がる台形."""
    t = q["torso"]
    nv, w = normals(t)[15], t[15]
    kl, kr = q["legL"][9], q["legR"][9]
    dn = (kl - kr) / (np.linalg.norm(kl - kr) + 1e-9)
    a, b = w + nv * 28, w - nv * 28
    if np.linalg.norm(a - kl) > np.linalg.norm(b - kl):
        a, b = b, a
    return [tuple(a), tuple(kl + dn * (LEG[9] + flare)), tuple(kr - dn * (LEG[9] + flare)), tuple(b)]


def draw_body(img, d, q, P, skin, top, bottom, shoes, sleeve_len, dress, hair, face):
    """ARM/LEG/TORSO の太さで人の体を描く. sleeve_len: 袖の長さ (0→24), dress: ワンピースか."""
    legs_col = skin if dress else bottom
    for k in ("legL", "legR"):
        ribbon(d, q[k], LEG, P(legs_col))
        oval(d, q[k][-1] + end_dir(q[k]) * 0.25, end_dir(q[k]), 34, 15, P(shoes, 0.4))
    if dress:
        d.polygon(skirt_poly(q), fill=P(top))
    for k in ("armL", "armR"):
        ribbon(d, q[k], ARM, P(skin))
        if sleeve_len > 0:
            ribbon(d, q[k][:sleeve_len + 1], ARM[:sleeve_len + 1] + 1.5, P(top))
        oval(d, q[k][-1] + end_dir(q[k]) * 0.2, end_dir(q[k]), 26, 15, P(skin))
    t = q["torso"]
    ribbon(d, t, TORSO, P(top))
    up = up_vector(q)
    neck_top = t[0] + up * 24
    ribbon(d, np.array([t[0] - up * 6, neck_top]), 9, P(skin))
    if face is not None:
        face(d, t, up)
    if hair is not None:
        paste_head(img, t[0] + up * 50, up, hair)
    return 140


# ------------------------------------------------------------------ シルエット
def draw_silhouette(img, d, q, who, P, outfit="suit"):
    ink = (34, 34, 44)

    def head(hd, m):
        if outfit == "dress":  # 長めの髪
            hd.ellipse((m - 38, m - 46, m + 38, m + 40), fill=P(ink))
            hd.rectangle((m - 36, m - 6, m + 36, m + 42), fill=P(ink))
        hd.ellipse((m - 31, m - 40, m + 31, m + 38), fill=P(ink))
        hd.ellipse((m - 34, m - 46, m + 34, m - 2), fill=P(ink))
    return draw_body(img, d, q, P, ink, ink, ink, ink, 24, outfit == "dress", head, None)


# ------------------------------------------------------------------ リアル寄り
SKIN = (240, 200, 168)
SUITS = [(44, 56, 86), (78, 78, 84), (102, 76, 60)]
DRESSES = [(214, 150, 162), (142, 172, 142), (122, 142, 192)]
HAIR = [(52, 38, 30), (78, 52, 36), (30, 26, 26)]
TIES = [(170, 40, 60), (60, 110, 160), (200, 160, 60)]


def draw_real(img, d, q, who, P, outfit="suit"):
    dress = outfit == "dress"
    top = DRESSES[who % 3] if dress else SUITS[who % 3]
    hair_c, skin = HAIR[who % 3], SKIN

    def shirt(dd, t, up):
        if dress:
            return
        nv = normals(t)
        a, b = t[1], t[8]
        dd.polygon([tuple(a + nv[1] * 15), tuple(b), tuple(a - nv[1] * 15)], fill=P((248, 248, 248)))
        dd.polygon([tuple(t[2] + nv[2] * 4), tuple(t[7] + nv[7] * 5), tuple(t[8]), tuple(t[7] - nv[7] * 5),
                    tuple(t[2] - nv[2] * 4)], fill=P(TIES[who % 3]))

    def head(hd, m):
        if dress:  # 後ろ髪
            hd.ellipse((m - 40, m - 48, m + 40, m + 36), fill=P(hair_c))
            hd.rectangle((m - 38, m - 6, m + 38, m + 46), fill=P(hair_c))
        hd.ellipse((m - 31, m - 40, m + 31, m + 38), fill=P(skin))
        hd.ellipse((m - 36, m - 6, m - 26, m + 12), fill=P(skin))  # 耳
        hd.ellipse((m + 26, m - 6, m + 36, m + 12), fill=P(skin))
        hd.chord((m - 34, m - 48, m + 34, m + 12), 180, 360, fill=P(hair_c))  # 前髪
        if not dress:
            hd.rectangle((m - 33, m - 18, m - 26, m + 2), fill=P(hair_c))
            hd.rectangle((m + 26, m - 18, m + 33, m + 2), fill=P(hair_c))
        for ex in (-12, 12):
            hd.ellipse((m + ex - 4, m - 2, m + ex + 4, m + 7), fill=P((40, 30, 30)))
            hd.line((m + ex - 8, m - 11, m + ex + 7, m - 12), fill=P(hair_c), width=3)
        hd.arc((m - 10, m + 12, m + 10, m + 24), 20, 160, fill=P((150, 70, 70)), width=3)
        hd.ellipse((m - 26, m + 6, m - 14, m + 14), fill=P((245, 175, 165)))  # ほお
        hd.ellipse((m + 14, m + 6, m + 26, m + 14), fill=P((245, 175, 165)))
    bottom = mix(top, (0, 0, 0), 0.15)
    return draw_body(img, d, q, P, skin, top, bottom, (30, 28, 30), 6 if dress else 24, dress, head, shirt)


def draw(style, img, d, q, who, tint=None, amt=0.0, outfit="suit"):
    """style で人を描き, 名札を置くための「首から上の高さ」を返す."""
    P = Paint(tint, amt)
    if style == "mannequin":
        return draw_mannequin(img, d, q, who, P)
    if style == "plush":
        return draw_plush(img, d, q, who, P)
    if style == "silhouette":
        return draw_silhouette(img, d, q, who, P, outfit)
    if style == "real":
        return draw_real(img, d, q, who, P, outfit)
    raise ValueError(style)
