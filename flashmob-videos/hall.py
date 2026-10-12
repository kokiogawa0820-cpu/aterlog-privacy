"""披露宴会場の見取り図 (席次表の配置どおり) と, 席から立ってダンスフロアへ集まる人の動き.

座標はワールド座標 (カメラ倍率 1 のとき画面のピクセルと同じ) で持ち, Camera で拡大・移動して描く.
"""
import colorsys
import math

import numpy as np

from common import INK, mix, text

# 画面上の会場エリア (左側). 右側は人数とテロップ用.
VIEW = (30, 125, 1390, 1065)
VIEW_C = ((VIEW[0] + VIEW[2]) / 2, (VIEW[1] + VIEW[3]) / 2)
PANEL_X, PANEL_W = 1655, 460

FLOOR_C = (710, 278)  # ダンスフロアの中心
FLOOR_RECT = (470, 205, 950, 340)
STAGE_RECT = (568, 138, 808, 192)  # 高砂 (新郎新婦の席)
TABLE_FILL = (236, 228, 212)
SEATED = (196, 186, 170)
HALL_BG = (247, 243, 234)
DOT_R = 9


def sheet_to_world(row, col):
    """席次表のセル位置 → 会場図の座標."""
    return np.array([50 + col * 5.75, 260 + (row - 15) * 11.8])


class Camera:
    def __init__(self, center=(710, 590), zoom=1.0):
        self.c = np.array(center, float)
        self.z = zoom

    def px(self, p):
        return (VIEW_C[0] + (p[0] - self.c[0]) * self.z, VIEW_C[1] + (p[1] - self.c[1]) * self.z)

    def lerp(self, other, u):
        return Camera(self.c + (other.c - self.c) * u, self.z + (other.z - self.z) * u)


def table_color(table):
    h = (ord(table) - ord("A")) * 0.137 % 1
    return tuple(int(v * 255) for v in colorsys.hsv_to_rgb(h, 0.6, 0.88))


class Hall:
    """会場の机と, 各ゲストの席の位置."""

    def __init__(self, guests, tables):
        self.guests = guests
        by_table = {}
        for i, g in enumerate(guests):
            by_table.setdefault(g.table, []).append(i)
        self.round_tables = {}  # 卓名 → 中心
        self.long_tables = {}  # 卓名 → (x, y0, y1)
        self.seat = np.zeros((len(guests), 2))
        for t, idx in by_table.items():
            cells = np.array([sheet_to_world(guests[i].row, guests[i].col) for i in idx])
            cols = sorted({guests[i].col for i in idx})
            if len(idx) > 12:  # 長テーブル (H, I): 席次表の並びそのまま, 机をはさんで2列
                cx = cells[:, 0].mean()
                self.long_tables[t] = (cx, cells[:, 1].min() - 18, cells[:, 1].max() + 18)
                for k, i in enumerate(idx):
                    side = -1 if guests[i].col == cols[0] else 1
                    self.seat[i] = (cx + side * 34, cells[k, 1])
            else:  # 丸テーブル: 席次表の2列 (左列は上から, 右列は下から) を円周に並べる
                center = self.round_center(tables[t])
                left = [i for i in idx if guests[i].col == cols[0]]
                right = [i for i in idx if guests[i].col != cols[0]][::-1]
                order = left + right
                n = max(len(order), 6)
                for k, i in enumerate(order):
                    a = math.pi * 0.5 + 2 * math.pi * (k + 0.5) / n  # 下から時計回り
                    self.seat[i] = center + 64 * np.array([math.cos(a), math.sin(a)])
                self.round_tables[t] = center

        for t, rc in tables.items():  # 空席だけの卓も机は描く
            if t not in self.round_tables and t not in self.long_tables:
                self.round_tables[t] = self.round_center(rc)

    @staticmethod
    def round_center(rc):
        """卓名のセル (行, 列) → 丸テーブルの中心. 席は卓名の下に約10行ぶん並んでいる."""
        return sheet_to_world(rc[0] + 10, rc[1])

    def nearest_to_floor(self, table):
        idx = [i for i, g in enumerate(self.guests) if g.table == table]
        return min(idx, key=lambda i: np.linalg.norm(self.seat[i] - FLOOR_C))

    def draw(self, d, cam, standing=()):
        """会場 (床, 高砂, 机, 座っている人) を描く. standing の人は席を空ける."""
        z = cam.z
        x0, y0 = cam.px(FLOOR_RECT[:2])
        x1, y1 = cam.px(FLOOR_RECT[2:])
        d.rounded_rectangle((x0, y0, x1, y1), radius=18 * z, fill=(230, 214, 186))
        sx0, sy0 = cam.px(STAGE_RECT[:2])
        sx1, sy1 = cam.px(STAGE_RECT[2:])
        d.rounded_rectangle((sx0, sy0, sx1, sy1), radius=10 * z, fill=(255, 255, 255), outline=INK, width=max(2, int(3 * z)))
        for k, who in enumerate(("新郎", "新婦")):
            p = cam.px((650 + 76 * k, 165))
            r = DOT_R * z * 1.2
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(120, 110, 100))
            text(d, (p[0] + (-1 if k == 0 else 1) * 34 * z, p[1]), who, 15 * z, fill=INK)
        text(d, cam.px((FLOOR_C[0], 318)), "ダンスフロア", 15 * z, fill=(150, 130, 100))
        for t, c in self.round_tables.items():
            p = cam.px(c)
            r = 46 * z
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=TABLE_FILL, outline=(205, 195, 178), width=max(1, int(2 * z)))
            text(d, p, t, 22 * z, fill=(150, 140, 125))
        for t, (cx, ya, yb) in self.long_tables.items():
            pa = cam.px((cx - 18, ya))
            pb = cam.px((cx + 18, yb))
            d.rounded_rectangle((pa[0], pa[1], pb[0], pb[1]), radius=8 * z, fill=TABLE_FILL, outline=(205, 195, 178), width=max(1, int(2 * z)))
            text(d, cam.px((cx, ya + 22)), t, 22 * z, fill=(150, 140, 125))
        standing = set(standing)
        for i in range(len(self.guests)):
            p = cam.px(self.seat[i])
            r = DOT_R * z * (0.55 if i in standing else 1)
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r),
                      fill=(232, 224, 210) if i in standing else SEATED)


def mask_panel(d):
    """拡大したときに会場図が右側 (人数・テロップ) にはみ出さないよう塗りつぶす."""
    d.rectangle((VIEW[2], 0, 1920, 1080), fill=HALL_BG)


def dance_spots(n, seed=4, spacing=22.0):
    """ダンスフロアに, きっちり並ばない程度にばらけた立ち位置を n 個作る (中心に近い順)."""
    rng = np.random.default_rng(seed)
    a, b = 330.0, 180.0
    pts = []
    tries = 0
    while len(pts) < n * 1.6 and tries < 60000:
        tries += 1
        p = np.array([FLOOR_C[0] + rng.uniform(-a, a), FLOOR_C[1] + rng.uniform(-b, b)])
        if ((p[0] - FLOOR_C[0]) / a) ** 2 + ((p[1] - FLOOR_C[1]) / b) ** 2 > 1 or p[1] < 205:
            continue
        if all(np.hypot(*(p - q)) >= spacing for q in pts):
            pts.append(p)
    pts = np.array(pts)
    rr = np.hypot((pts[:, 0] - FLOOR_C[0]) / 1.9, pts[:, 1] - FLOOR_C[1]) * (1 + rng.uniform(-0.15, 0.15, len(pts)))
    pts = list(pts[np.argsort(rr)])
    # 最初の数人は名札が重ならないよう, 少し間をあけた場所から使う
    first, rest = [], []
    for p in pts:
        if len(first) < 12 and all(np.hypot(*(p - q)) >= 52 for q in first):
            first.append(p)
        else:
            rest.append(p)
    return np.array(first + rest)[:n]


class Mover:
    """席から立ち上がり, 少し曲がった道すじでフロアへ行き, 着いたらみんなで同じ振りで踊る."""

    def __init__(self, seat, spot, t_start, seed):
        rng = np.random.default_rng(seed)
        self.p0 = np.array(seat, float)
        self.p1 = np.array(spot, float)
        dist = np.linalg.norm(self.p1 - self.p0)
        self.t0 = t_start
        self.stand = 0.25  # 立ち上がる時間
        self.dur = float(np.clip(dist / 650, 0.5, 1.3))
        v = self.p1 - self.p0
        nrm = np.array([-v[1], v[0]]) / (dist + 1e-9)
        self.ctrl = (self.p0 + self.p1) / 2 + nrm * dist * rng.uniform(-0.25, 0.25)

    def arrive(self):
        return self.t0 + self.stand + self.dur

    def pos(self, t):
        if t < self.t0 + self.stand:
            return self.p0
        u = min(1.0, (t - self.t0 - self.stand) / self.dur)
        u = u * u * (3 - 2 * u)
        p = (1 - u) ** 2 * self.p0 + 2 * (1 - u) * u * self.ctrl + u ** 2 * self.p1
        if t > self.arrive():  # 全員そろった振り付け (左右ステップ)
            k = min(1.0, (t - self.arrive()) / 0.3)
            beat = 2 * math.pi * 1.6 * t
            p = p + k * np.array([6 * math.sin(beat), -4 * abs(math.sin(beat))])
        return p


def draw_movers(d, cam, movers, colors, t, tags=None):
    z = cam.z
    items = []
    for i, m in movers.items():
        if t < m.t0:
            continue
        items.append((m.pos(t)[1], i))
    for _, i in sorted(items):
        m = movers[i]
        p = cam.px(m.pos(t))
        grow = min(1.0, (t - m.t0) / m.stand)
        r = DOT_R * z * (1 + 0.25 * grow)
        d.ellipse((p[0] - r + 2 * z, p[1] - r + 3 * z, p[0] + r + 2 * z, p[1] + r + 3 * z), fill=(205, 195, 178))
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=mix(SEATED, colors[i], grow), outline=INK,
                  width=max(1, int(2 * z)))
        if tags and i in tags:
            s = 17 * z
            text(d, (p[0], p[1] - r - s * 0.9), tags[i], s, fill=INK, stroke=max(2, int(3 * z)))
