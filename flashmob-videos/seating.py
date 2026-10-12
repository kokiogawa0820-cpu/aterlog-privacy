"""席次表 (.xlsx) を読み込む.

席次表は「続柄」「ふりがな」「名前 + 様」が縦に並ぶ形式 (式場の席次表テンプレート).
名前のセルの右に「様」があるセルを名前とみなし, 2つ上を続柄, 1つ上をふりがなとして読む.
卓 (テーブル) は, 名前より上にある一番近い1文字のアルファベット (卓名) で決める.

席次表には参列者の個人情報が入っているので, リポジトリには入れない (data/ は .gitignore 済み).
"""
import os
from dataclasses import dataclass

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seating.xlsx")
MAIN_ROWS = 85  # これより下は式場の記入例・注意書き


@dataclass
class Guest:
    name: str  # 例: "山田 太郎"
    role: str  # 例: "新郎新婦先輩"
    table: str  # 例: "A"
    row: int  # 席次表のセル位置 (会場図の配置に使う)
    col: int

    @property
    def surname(self):
        parts = self.name.split()
        return parts[0] if len(parts) > 1 else self.name[:2]


def load(path=DEFAULT_PATH):
    """(ゲストの一覧, {卓名: 卓名セルの (行, 列)}) を返す."""
    import openpyxl

    ws = openpyxl.load_workbook(path, data_only=True).active
    grid = {(c.row, c.column): c.value for row in ws.iter_rows() for c in row if c.value not in (None, "")}
    heads = [(r, c, v) for (r, c), v in grid.items()
             if r <= MAIN_ROWS and isinstance(v, str) and len(v) == 1 and v.isascii() and v.isalpha()]
    guests = []
    for (r, c), v in sorted(grid.items()):
        if r > MAIN_ROWS or not isinstance(v, str) or v == "様" or "\n" in v:
            continue
        if not any(grid.get((r, c + k)) == "様" for k in range(1, 13)):
            continue
        role = grid.get((r - 2, c))
        # 続柄の欄にふりがなが入っているなど, 続柄らしくないものは空にする
        if not isinstance(role, str) or not any(k in role for k in ("新郎", "新婦")):
            role = ""
        cand = [h for h in heads if h[0] <= r and abs(h[1] - c) <= 16]
        table = max(cand)[2] if cand else "?"
        guests.append(Guest(" ".join(v.replace("　", " ").split()), role, table, r, c))
    tables = {v: (r, c) for r, c, v in sorted(heads)}
    return guests, tables


def find(guests, name):
    key = name.replace(" ", "").replace("　", "")
    for g in guests:
        if g.name.replace(" ", "") == key:
            return g
    raise KeyError(f"席次表に {name} さんが見つかりません")
