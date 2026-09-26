"""東大のイチョウマーク画像から, 色ごとにきれいに塗り直した高解像度 PNG (透過) を作る.

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
    Image.fromarray(rgba).resize((S, S), Image.LANCZOS).save(out)
    print("wrote", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    trace(args[0], *map(float, args[1:4]))
