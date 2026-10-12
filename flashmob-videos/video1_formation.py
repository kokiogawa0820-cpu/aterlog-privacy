"""動画1: 席次表どおりの会場で, ゲストが1人ずつ席から立ってダンスフロアへ → どんどん加速して全員に.
(ツッコミ「参列者全員参加させるんか！」は撮影側で. --tsukkomi を付けると文字のツッコミも入れる)

使い方: python3 video1_formation.py [出力.mp4] [--seating 席次表.xlsx] [--preview] [--tsukkomi]
"""
import sys

import numpy as np

from common import INK, Audio, caption, clamp01, ease, header, mix, render, text, tsukkomi
import hall
import seating

TITLE = "フラッシュモブ 当日の流れ"
SLOW_TABLES = ["H", "I", "B", "C"]  # 最初にゆっくり紹介する人の卓 (各卓でフロアに一番近い人)
NAMED_FAST = 8  # その次に名前を出しながら加速する人数


def order_guests(h):
    """登場順: ゆっくり紹介する4人 → フロアに近い人から順に (少しばらつかせる)."""
    first = [h.nearest_to_floor(t) for t in SLOW_TABLES if any(g.table == t for g in h.guests)]
    rng = np.random.default_rng(11)
    rest = [i for i in range(len(h.guests)) if i not in first]
    dist = {i: np.linalg.norm(h.seat[i] - hall.FLOOR_C) * rng.uniform(0.8, 1.2) for i in rest}
    return first + sorted(rest, key=dist.get)


def start_times(n, t0, slow):
    """立ち上がる時刻. 最初はゆっくり, そこから等比的に加速 (最速で1フレームに4人)."""
    times, t, dt = [], t0, 1.7
    for k in range(n):
        times.append(t)
        if k == slow - 1:
            dt = 0.75
        elif k >= slow:
            dt = max(1 / 30 / 4, dt * 0.86)
        t += dt
    return np.array(times)


def draw_counter(d, n, bump):
    x, y1, y2 = hall.PANEL_X, 170, 420
    d.rounded_rectangle((x - 210, y1, x + 210, y2), radius=26, fill=(255, 255, 255), outline=INK, width=5)
    text(d, (x, y1 + 50), "配置人数", 46, fill=(110, 100, 90))
    text(d, (x, y1 + 160), f"{n}人", 120 * (1 + 0.15 * bump), fill=mix(INK, (210, 30, 40), clamp01((n - 40) / 110)))


def cap(d, s, **kw):
    caption(d, s, x=hall.PANEL_X, y=640, width=hall.PANEL_W, size=46, **kw)


def build(out_path, seating_path=seating.DEFAULT_PATH, with_tsukkomi=False, previews=()):
    guests, tables = seating.load(seating_path)
    h = hall.Hall(guests, tables)
    order = order_guests(h)
    n = len(order)
    slow = len([t for t in SLOW_TABLES if any(g.table == t for g in guests)])
    spots = hall.dance_spots(n)
    t_title = 2.4
    t_zoom = 1.0
    times = start_times(n, t_title + t_zoom, slow)
    colors = {i: hall.table_color(guests[i].table) for i in range(len(guests))}
    movers = {i: hall.Mover(h.seat[i], spots[k], times[k], seed=k) for k, i in enumerate(order)}
    t_all = max(m.arrive() for m in movers.values()) + 0.4
    hold = 3.0

    wide = hall.Camera((710, 590), 1.0)
    close = hall.Camera((700, 330), 1.9)

    def camera(t):
        if t < t_title:
            return wide
        if t < t_title + t_zoom:
            return wide.lerp(close, ease((t - t_title) / t_zoom))
        # 立つ人が増えてきたら, 会場全体が見えるまで引く
        k = np.searchsorted(times, t)
        return close.lerp(wide, ease((k - slow) / 22))

    audio = Audio(t_all + hold + 5)
    scenes = []

    def frame(img, d, t):
        cam = camera(t)
        started = [i for i in order if movers[i].t0 <= t]
        h.draw(d, cam, standing=started)
        tags = {i: guests[i].surname for i in order[:slow + NAMED_FAST]}
        hall.draw_movers(d, cam, movers, colors, t, tags=tags if cam.z > 1.25 else None)
        hall.mask_panel(d)
        header(d, TITLE)
        return len(started)

    def title(img, d, t, u):
        frame(img, d, t)
        cap(d, "会場の席次です", pop=t / 0.3)

    scenes.append((t_title, title))
    audio.pop(0.05, 520, 0.2)

    last = {"n": 0, "t": -9.0, "cap": None, "cap_t": -9.0}

    def placing(img, d, lt, u):
        t = t_title + lt
        cnt = frame(img, d, t)
        if cnt != last["n"]:
            last["n"], last["t"] = cnt, t
        if cnt == 0:
            cap(d, "ダンスフロアに\n集まります", pop=lt / 0.3)
            return
        draw_counter(d, cnt, max(0.0, 1 - (t - last["t"]) / 0.15))
        k = cnt - 1
        g = guests[order[k]]
        sub = f"（{g.role}・{g.table}卓）" if g.role else f"（{g.table}卓）"
        if k < slow:
            cap(d, ("まず\n" if k == 0 else "続いて\n") + f"{g.name}さん", sub=sub, pop=(t - times[k]) / 0.25)
        else:  # 速くなってからは, 読める程度にテロップの切り替えを間引く
            if t - last["cap_t"] >= 0.2 or last["cap"] is None:
                last["cap"], last["cap_t"] = (f"{g.name}さん", sub), t
            cap(d, last["cap"][0], sub=last["cap"][1])

    scenes.append((t_all - t_title, placing))

    prev = -1.0
    for k, tk in enumerate(times):  # 立ち上がる音. 速くなったら間引いて, 音程を少しずつ上げる
        if tk - prev >= 0.07:
            audio.pop(tk, 520 * 1.04 ** min(k, 20), 0.25) if k < 12 else audio.tick(tk, 900 + 6 * k, 0.18)
            prev = tk

    def final(img, d, lt, u):
        t = t_all + lt
        frame(img, d, t)
        draw_counter(d, n, 0)
        cap(d, f"以上 {n}名で\n踊ります", pop=lt / 0.3)

    scenes.append((hold, final))
    audio.chime(t_all, 0.2)

    if with_tsukkomi:
        def punch(img, d, lt, u):
            final(img, d, hold + lt, 1)
            tsukkomi(img, lt / 2.5, "参列者全員 参加させるんか！", base_size=120)

        scenes.append((2.6, punch))
        audio.don(t_all + hold)

    total = render(scenes, out_path, audio, preview_times=previews)
    print(f"wrote {out_path} ({total:.1f}s, {n}名)")


def arg_value(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


if __name__ == "__main__":
    seat_path = arg_value("--seating", seating.DEFAULT_PATH)
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--seating"]
    out = args[0] if args else "video1_formation.mp4"
    build(out, seat_path, with_tsukkomi="--tsukkomi" in sys.argv,
          previews=[1.0, 3.0, 4.5, 6.5, 9.5, 11.5, 13.0, 15.0, 18.0] if "--preview" in sys.argv else [])
