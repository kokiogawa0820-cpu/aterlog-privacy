# フラッシュモブ動画（ツッコミ前フリ用）

横長 16:9 の動画（1920×1080 / 30fps / 効果音付き）を Python で生成します。
ツッコミは撮影側で入れる前提なので、どれも最後に数秒間の「間」を残して終わります。

| ファイル | 長さ | 内容 | ツッコミ（撮影側） |
|---|---|---|---|
| `video1_formation.mp4` | 約20秒 | 席次表どおりの会場図。「会場の席次です」→ ダンスフロアに寄って、ゲストが1人ずつ自分の席から立ってフロアへ（「まず 〇〇さん（続柄・卓）」「続いて〜」）→ どんどん加速し、会場全体に引きながら全員が席を立ってフロアに集まり、着いた人からそろって踊る → 右上の配置人数が全員（163人）に →「以上 163名で踊ります」 | 参列者全員参加させるんか！ |
| `video2_ginkgo.mp4` | 約21秒 | 席次表から選んだ3人（名札付き）のピクトグラム。何を作るかは伏せたまま淡々と説明 → 伸び縮み・逆関節・巻きつき → ゆっくり色が付いて東大マークが浮き出る（`--long` で普通の動きを増やした長い版） | 無理やん！ |
| `video3_combined.mp4` | 約32秒 | 1本だけで見せる版。会場図で選ばれた3人が自分の席から立ってフロアへ →「この3名で1つの形を作ります」→ フロアに寄りながら動画2へつながる（`--long` で長い版） | 無理やん！ |

## 席次表

- 席次表（.xlsx）は `data/seating.xlsx` に置きます。別の場所なら `--seating パス` で指定します。参列者の個人情報なので `data/` と、名前入りの書き出し動画（`*.mp4`）はリポジトリに入れません（`.gitignore` 済み）。
- 読み込みは `seating.py`。名前の右に「様」があるセルを名前、その2つ上を続柄として読み、卓は上にある卓名（A〜R）で決めます。会場図（`hall.py`）は、席次表の卓名の位置をそのまま会場の配置にしています。
- 動画2・3の3人は `data/picks.txt`（1行に1人、左から。なければ A・C・B 卓の先頭の人）、動画1で最初にゆっくり紹介する人は `video1_formation.py` の `SLOW_TABLES`（各卓でフロアに一番近い人）で変えられます。

## 再生成

```bash
pip install pillow numpy imageio imageio-ffmpeg
pip install openpyxl
python3 video1_formation.py                   # → video1_formation.mp4
python3 video2_ginkgo.py                      # → video2_ginkgo.mp4
python3 video2_ginkgo.py --long               # → video2_ginkgo_long.mp4（長い版）
python3 video3_combined.py                    # → video3_combined.mp4
python3 video1_formation.py --tsukkomi        # 動画1の最後に文字のツッコミ演出も入れる場合
python3 video1_formation.py out.mp4 --preview # 確認用の静止画 PNG も書き出す
```

日本語フォントは IPA ゴシックを使っています（パスは `common.py` の `FONT_PATHS` で変更できます）。

## 東大のマークについて（動画 2）

- `assets/utokyo_mark.png`（マークのみ）と `assets/utokyo_logo.png`（下に「東京大学」の文字付き。右側に出すもの）は、いただいたマーク画像を `trace_mark.py` で色ごとに塗り直して高解像度にしたものです。人がなぞる形（`video2_ginkgo.py` の `pose_yellow` / `pose_blue` / `pose_ring`）も、この画像の座標に合わせてあります。
- 別の画像から作り直すときは `python3 trace_mark.py 画像.jpg 中心x 中心y 半径` を実行します。
- 東京大学のマークは大学の登録商標です。公開する範囲や使い方には気をつけてください（このリポジトリが公開リポジトリの場合、画像もそのまま公開されます）。
