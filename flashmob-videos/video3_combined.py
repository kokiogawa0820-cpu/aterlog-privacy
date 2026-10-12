"""動画3 (1本だけで見せる版): 席次表どおりの会場から3人が立ってダンスフロアへ → そのまま動画2 (イチョウ) へ.

使い方: python3 video3_combined.py [出力.mp4] [--seating 席次表.xlsx] [--long] [--preview]
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

from common import BG, H, W, Audio, caption, ease, header, render
import hall
import seating
import video2_ginkgo as v2


def cap(d, s, **kw):
    caption(d, s, x=hall.PANEL_X, y=600, width=hall.PANEL_W, size=46, **kw)


def build(out_path, seating_path=seating.DEFAULT_PATH, long=False, previews=()):
    guests, tables = seating.load(seating_path)
    h = hall.Hall(guests, tables)
    picks = v2.choose_picks(guests)
    idx = [guests.index(g) for g in picks]
    spots = [(600, 270), (710, 270), (820, 270)]  # 動画2の並び (左から) と同じ順にフロアへ

    t_title, t_zoom, each, t_wait, t_trans = 2.2, 1.0, 2.0, 1.2, 1.2
    t_starts = [t_title + t_zoom + k * each for k in range(3)]
    movers = {i: hall.Mover(h.seat[i], spots[k], t_starts[k], seed=k) for k, i in enumerate(idx)}
    colors = {i: hall.table_color(guests[i].table) for i in idx}
    tags = {i: g.surname for i, g in zip(idx, picks)}
    t_floor = t_starts[-1] + each + t_wait

    wide = hall.Camera((710, 590), 1.0)
    mid = hall.Camera((700, 380), 1.3)
    close = hall.Camera((710, 270), 3.2)

    audio = Audio(80)
    scenes = []

    def hall_frame(img, d, t, cam):
        h.draw(d, cam, standing=[i for i in idx if movers[i].t0 <= t])
        hall.draw_movers(d, cam, movers, colors, t, tags=tags)
        hall.mask_panel(d)
        header(d, v2.TITLE)

    def title(img, d, t, u):
        hall_frame(img, d, t, wide)
        cap(d, "会場の席次です", pop=t / 0.3)

    scenes.append((t_title, title))
    audio.pop(0.05, 520, 0.2)

    def calling(img, d, lt, u):
        t = t_title + lt
        cam = wide.lerp(mid, ease(lt / t_zoom))
        hall_frame(img, d, t, cam)
        k = int(np.searchsorted(t_starts, t, side="right")) - 1
        if k < 0:
            cap(d, "3名に前へ\n出ていただきます", pop=lt / 0.3)
        elif t < t_starts[-1] + each:
            g = picks[k]
            cap(d, f"{g.name}さん", sub=f"（{g.role}・{g.table}卓）", pop=(t - t_starts[k]) / 0.25)
        else:
            cap(d, "この3名で\n1つの形を作ります", pop=(t - t_starts[-1] - each) / 0.3)

    scenes.append((t_floor - t_title, calling))
    for ts in t_starts:
        audio.pop(ts, 600, 0.22)
    audio.pop(t_starts[-1] + each, 660, 0.18)

    # 動画2のシーン (フロアの3人 → ピクトグラムの3人)
    t_v2 = t_floor + t_trans
    v2_scenes, _ = v2.make_scenes(audio, t0=t_v2, long=long, picks=picks, intro_dur=1.6)

    def transition(img, d, lt, u):
        """フロアに寄りながら, 動画2の最初の画面へ切り替える."""
        t = t_floor + lt
        hall_frame(img, d, t, mid.lerp(close, ease(u)))
        cap(d, "この3名で\n1つの形を作ります")
        k = ease(max(0.0, (u - 0.45) / 0.55))
        if k > 0:
            nxt = Image.new("RGB", (W, H), BG)
            v2_scenes[0][1](nxt, ImageDraw.Draw(nxt), 0.0, 0.0)
            img.paste(Image.blend(img, nxt, k))

    scenes.append((t_trans, transition))
    scenes += v2_scenes

    total = render(scenes, out_path, audio, preview_times=previews)
    print(f"wrote {out_path} ({total:.1f}s)")


if __name__ == "__main__":
    seat_path = sys.argv[sys.argv.index("--seating") + 1] if "--seating" in sys.argv else seating.DEFAULT_PATH
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--seating"]
    long = "--long" in sys.argv
    out = args[0] if args else ("video3_combined_long.mp4" if long else "video3_combined.mp4")
    build(out, seat_path, long=long,
          previews=[1.0, 3.0, 4.5, 6.5, 8.5, 10.0, 11.8, 12.6, 14.0, 20.0, 27.0, 31.0] if "--preview" in sys.argv else [])
