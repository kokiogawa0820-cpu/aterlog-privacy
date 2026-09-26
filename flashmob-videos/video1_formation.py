"""動画1: ◯(=人)を紹介しながら配置 → 激ムズフォーメーション移動 → 「無理やん！」"""
import colorsys
import math
import sys
from collections import deque

import numpy as np
from PIL import ImageDraw

from common import (BG, INK, W, Audio, caption, clamp01, ease, ease_out_back, font, header, mix, render,
                    text, tsukkomi)

N = 16
NAMES = ["田中", "佐藤", "鈴木", "高橋", "伊藤", "渡辺", "山本", "中村",
         "小林", "加藤", "吉田", "山田", "佐々木", "山口", "松本", "井上"]
COLORS = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.62, 0.9)) for i in range(N)]
CX, CY, UNIT = 540, 870, 400  # 舞台中心とスケール (単位座標 1.0 = 400px)
R = 40
TITLE = "フラッシュモブ振付会議"

HOME = np.array([[(-0.66 + 0.44 * (i % 4)), (-0.66 + 0.44 * (i // 4))] for i in range(N)])
# 紹介順に目立つ位置へ: 田中さんはセンター前
HOME[[0, 5]] = HOME[[5, 0]]
HOME[0] = [0.0, 0.25]
HOME[5] = [-0.22, -0.66]


def to_px(p):
    return CX + p[0] * UNIT, CY + p[1] * UNIT


def draw_stage(d):
    d.rounded_rectangle((60, 400, 1020, 1340), radius=24, fill=(233, 224, 206), outline=INK, width=4)
    for k in range(1, 8):
        x = 60 + k * 120
        d.line((x, 400, x, 1340), fill=(218, 207, 186), width=2)
        d.line((60, 280 + k * 120 + 120, 1020, 280 + k * 120 + 120), fill=(218, 207, 186), width=2)
    d.line((CX - 20, CY, CX + 20, CY), fill=(200, 60, 60), width=4)  # バミリ
    d.line((CX, CY - 20, CX, CY + 20), fill=(200, 60, 60), width=4)
    text(d, (CX, 1385), "▼ 客席 ▼", 38, fill=(120, 110, 95))


def draw_people(d, pos, scales, trails=None, spaghetti=None, spaghetti_fade=0.0):
    if spaghetti is not None and spaghetti_fade > 0:
        for i in range(N):
            pts = [to_px(p) for p in spaghetti[i]]
            d.line(pts, fill=mix(BG, COLORS[i], 0.85 * spaghetti_fade), width=4, joint="curve")
    if trails is not None:
        for i in range(N):
            tr = list(trails[i])
            for k in range(1, len(tr)):
                a = k / len(tr)
                d.line((to_px(tr[k - 1]), to_px(tr[k])), fill=mix((233, 224, 206), COLORS[i], a * 0.8),
                       width=int(4 + 18 * a))
    order = sorted(range(N), key=lambda i: pos[i][1])
    for i in order:
        s = scales[i]
        if s <= 0:
            continue
        x, y = to_px(pos[i])
        r = R * s
        d.ellipse((x - r + 4, y - r + 8, x + r + 4, y + r + 8), fill=(200, 190, 170))  # 影
        d.ellipse((x - r, y - r, x + r, y + r), fill=COLORS[i], outline=INK, width=4)
        if s > 0.5:
            text(d, (x, y), NAMES[i], (22 if len(NAMES[i]) > 2 else 27) * s, fill=(255, 255, 255),
                 stroke=3, stroke_fill=INK)


# ----------------------------------------------------------------- ダンスの各パート
def ring(u, i, r_out, r_in, rot):
    outer = i % 2 == 0
    k = i // 2
    r = r_out + (r_in - r_out) * ease(u) if outer else r_in + (r_out - r_in) * ease(u)
    a = 2 * math.pi * k / 8 + (rot if outer else -rot) * u + (0 if outer else math.pi / 8)
    return [r * math.cos(a), r * math.sin(a)]


def seg_rings(u):
    return np.array([ring(u, i, 0.85, 0.35, 2 * math.pi) for i in range(N)])


def seg_figure8(u):
    out = []
    for i in range(N):
        if i == 0:
            s = 2 * math.pi * 2 * u
            out.append([0.75 * math.sin(s), 0.45 * math.sin(2 * s)])
        else:
            a = 2 * math.pi * i / N - 3 * math.pi * u
            r = 0.95 - 0.6 * u
            out.append([r * math.cos(a), r * math.sin(a)])
    return np.array(out)


def seg_spiro(u):
    out = []
    for i in range(N):
        a = 2 * math.pi * i / N + 2 * math.pi * u
        k = 3 + i % 5
        b = 2 * math.pi * k * u * 1.5 + i
        out.append([0.5 * math.cos(a) + 0.28 * math.cos(b), 0.5 * math.sin(a) + 0.28 * math.sin(b)])
    return np.array(out)


def seg_swap(u):
    """2拍ごとに隣とすり抜け (4回)."""
    base = np.array([[-0.75 + 0.5 * (i % 4), -0.6 + 0.4 * (i // 4)] for i in range(N)])
    beat = min(3, int(u * 4))
    lu = ease(u * 4 - beat)
    pos = base.copy()
    for i in range(N):
        # 奇数拍: 左右ペア, 偶数拍: 上下ペア
        if beat % 2 == 0:
            partner = i ^ 1
        else:
            partner = i + 4 if (i // 4) % 2 == 0 else i - 4
        src = base[i] if (beat // 2) % 2 == 0 else base[partner]
        dst = base[partner] if (beat // 2) % 2 == 0 else base[i]
        pos[i] = src + (dst - src) * lu
    return pos


def heart_pt(s):
    x = 16 * math.sin(s) ** 3
    y = -(13 * math.cos(s) - 5 * math.cos(2 * s) - 2 * math.cos(3 * s) - math.cos(4 * s))
    return [x / 17 * 0.85, y / 17 * 0.85 - 0.05]


def seg_heart(u):
    return np.array([heart_pt(2 * math.pi * (i / N + 0.5 * u)) for i in range(N)])


def star_pt(s):
    s = s % 1
    k = int(s * 10)
    f = s * 10 - k

    def v(j):
        a = -math.pi / 2 + j * math.pi / 5
        r = 0.9 if j % 2 == 0 else 0.38
        return np.array([r * math.cos(a), r * math.sin(a)])

    return v(k) + (v(k + 1) - v(k)) * f


def seg_star(u):
    return np.array([star_pt(i / N - 0.6 * u) for i in range(N)])


def seg_collapse(u):
    return np.zeros((N, 2)) + 0.0


def seg_burst(u):
    rng = np.random.default_rng(3)
    shuffle = rng.permutation(N)
    out = []
    for i in range(N):
        a = 2 * math.pi * i / N
        burst = np.array([1.25 * math.cos(a), 1.25 * math.sin(a)])
        if u < 0.45:
            out.append(burst * ease(u / 0.45))
        else:
            out.append(burst + (HOME[shuffle[i]] - burst) * ease((u - 0.45) / 0.55))
    return np.array(out)


SEGMENTS = [
    # (秒, パート関数, テロップ, 補足)
    (2.4, seg_rings, "外周と内周が\n逆回転しながら入れ替わり", "（内外で半拍ずらす）"),
    (2.2, seg_figure8, "田中さんだけ8の字\n他は反時計回りの螺旋", "（田中さんは2周）"),
    (2.0, seg_spiro, "全員それぞれ違う\nスピログラフを描いて", "（各自で計算しておいてください）"),
    (1.8, seg_swap, "2拍ごとに\n隣の人とすり抜け", "（ぶつかりますが気合いで）"),
    (1.5, seg_heart, "ハートを描きながら回転", None),
    (1.3, seg_star, "そのまま星に", None),
    (1.0, seg_collapse, "全員センターの\n1点に重なって", "（物理的に）"),
    (1.4, seg_burst, "放射状に散開して\n全員位置シャッフル", None),
]


def build(out_path, previews=()):
    audio = Audio(60)
    scenes = []
    t_cursor = [0.0]
    state = {"pos": HOME.copy(), "scales": np.zeros(N)}

    def add(dur, fn, sound=None):
        if sound:
            sound(t_cursor[0])
        scenes.append((dur, fn))
        t_cursor[0] += dur

    # 1) タイトル
    def title(img, d, t, u):
        header(d, TITLE)
        draw_stage(d)
        a = ease_out_back(t / 0.5)
        d.rounded_rectangle((110, 700, 970, 1040), radius=30, fill=(255, 255, 255), outline=INK, width=5)
        text(d, (W / 2, 870), "フォーメーション\n説明会", int(96 * (0.5 + 0.5 * a)))
        caption(d, "本番は来週です", pop=t / 0.3)

    add(2.4, title, lambda t: audio.pop(t + 0.05, 440))

    # 2) ◯を1人ずつ紹介
    intros = [
        (1.7, [0], "これが田中さん", "（センター）"),
        (1.7, [1], "こちら佐藤さん", "（田中さんの幼なじみ）"),
        (1.7, [2], "鈴木さん", "（今日が初参加）"),
        (1.7, [3], "高橋さん", "（言い出しっぺ）"),
    ] + [(0.38, [i], f"{NAMES[i]}さん", None) for i in range(4, 10)] + [
        (1.8, list(range(10, 16)), "あと6人", "（名前は各自覚えてください）"),
    ]
    for k, (dur, ids, cap, sub) in enumerate(intros):
        def fn(img, d, t, u, ids=ids, cap=cap, sub=sub):
            header(d, TITLE)
            draw_stage(d)
            sc = state["scales"].copy()
            for i in ids:
                sc[i] = ease_out_back(t / 0.3)
            draw_people(d, HOME, sc)
            if len(ids) == 1:
                x, y = (to_px(HOME[ids[0]]))
                d.line((x, y - R - 12, x, y - R - 60), fill=(210, 30, 40), width=6)
                d.polygon([(x - 14, y - R - 26), (x + 14, y - R - 26), (x, y - R - 6)], fill=(210, 30, 40))
            caption(d, cap, sub=sub, pop=t / 0.25)
            if u >= 0.999:
                state["scales"][list(ids)] = 1

        add(dur, fn, (lambda t, k=k: audio.pop(t, 520 * 1.06 ** k)))

    # 3) 基本の形 (簡単)
    V = np.array([[(i - 7.5) * 0.11, -0.55 + abs(i - 7.5) * 0.13] for i in range(N)])
    order = np.argsort(HOME[:, 0] + HOME[:, 1] * 0.01)
    V_assign = np.zeros((N, 2))
    V_assign[order] = V

    def basic(img, d, t, u):
        header(d, TITLE)
        draw_stage(d)
        pos = HOME + (V_assign - HOME) * ease(u * 1.4)
        draw_people(d, pos, np.ones(N))
        caption(d, "まずは基本のV字", sub="（ここまではOK）" if u > 0.6 else None, pop=t / 0.25)

    add(2.6, basic, lambda t: audio.whoosh(t + 0.2, 0.8, 0.2))

    def ready(img, d, t, u):
        header(d, TITLE)
        draw_stage(d)
        draw_people(d, V_assign, np.ones(N))
        caption(d, "では本番の動きです", sub="（サビ前まで）", pop=t / 0.25)

    add(1.4, ready)

    # 4) 激ムズ移動. 各パートの開始位置の差分は最初の1/3で吸収して連続にする
    history = [[] for _ in range(N)]
    trails = [deque(maxlen=10) for _ in range(N)]
    prev_end = V_assign.copy()
    plan = SEGMENTS + [(dur * 0.33, fn, None, None) for dur, fn, _, _ in SEGMENTS]  # 最後に3倍速でもう1回
    for idx, (dur, seg, cap, sub) in enumerate(plan):
        start_off = prev_end - seg(0.0)
        fast = idx >= len(SEGMENTS)

        def fn(img, d, t, u, seg=seg, cap=cap, sub=sub, start_off=start_off, fast=fast):
            header(d, TITLE)
            draw_stage(d)
            pos = seg(u) + start_off * (1 - ease(u * 3))
            for i in range(N):
                history[i].append(pos[i].copy())
                trails[i].append(pos[i].copy())
            draw_people(d, pos, np.ones(N), trails=trails)
            if fast:
                caption(d, "これを3倍速でもう1回！", size=62)
                d.text((W - 60, 230), "×3", font=font(90), fill=(210, 30, 40), anchor="rm")
            else:
                caption(d, cap, sub=sub, size=58, pop=t / 0.2)

        e = seg(1.0)
        prev_end = e + start_off * 0  # 終点 (オフセットは u=1 で 0)

        def sfx(t0, dur=dur, fast=fast, seg=seg):
            beats = 4 if not fast else 2
            for b in range(beats):
                audio.tick(t0 + dur * b / beats, 1800 if b == 0 else 1300)
            if seg is seg_collapse:
                audio.sweep(t0, 900, 200, dur * 0.9, 0.2)
            if seg is seg_burst and not fast:
                audio.whoosh(t0, 0.5, 0.35)

        add(dur, fn, sfx)

    final_pos = prev_end

    # 5) 全員の軌跡 (スパゲッティ)
    def summary(img, d, t, u):
        header(d, TITLE)
        draw_stage(d)
        draw_people(d, final_pos, np.ones(N), spaghetti=history, spaghetti_fade=ease(t / 0.6))
        caption(d, "以上です！\nちなみに本番は来週です", sub="（全員の動線はこちら）", size=58, pop=t / 0.25)

    add(3.0, summary, lambda t: audio.chime(t))

    # 6) ツッコミ
    def punch(img, d, t, u):
        header(d, TITLE)
        draw_stage(d)
        draw_people(d, final_pos, np.ones(N), spaghetti=history, spaghetti_fade=1)
        tsukkomi(img, t / 2.5)

    add(2.6, punch, lambda t: audio.don(t))

    total = render(scenes, out_path, audio, preview_times=previews)
    print(f"wrote {out_path} ({total:.1f}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "video1_formation.mp4"
    build(out, previews=[1.5, 5.0, 12.0, 20.0, 25.0, 30.0, 37.0, 39.5] if "--preview" in sys.argv else [])
