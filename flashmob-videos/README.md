# フラッシュモブ動画（ツッコミ前フリ用）

横長 16:9 の動画（1920×1080 / 30fps / 効果音付き）を Python で生成します。
ツッコミは撮影側で入れる前提なので、どれも最後に数秒間の「間」を残して終わります。

| ファイル | 長さ | 内容 | ツッコミ（撮影側） |
|---|---|---|---|
| `video1_formation.mp4` | 約20秒 | 席次表どおりの会場図。「会場の席次です」→ ダンスフロアに寄って、ゲストが1人ずつ自分の席から立ってフロアへ（「まず 〇〇さん（続柄・卓）」「続いて〜」）→ どんどん加速し、会場全体に引きながら全員が席を立ってフロアに集まり、着いた人からそろって踊る → 右上の配置人数が全員（163人）に →「以上 163名で踊ります」 | 参列者全員参加させるんか！ |
| `video2_ginkgo.mp4` | 約21秒 | 席次表から選んだ3人（名札付き）のピクトグラム。何を作るかは伏せたまま淡々と説明 → 伸び縮み・逆関節・巻きつき → ゆっくり色が付いて東大マークが浮き出る（`--long` で普通の動きを増やした長い版） | 無理やん！ |
| `video3_combined.mp4` | 約32秒 | 1本だけで見せる版。会場図で選ばれた3人が自分の席から立ってフロアへ →「この3名で1つの形を作ります」→ フロアに寄りながら動画2へつながる（`--long` で長い版） | 無理やん！ |

## 人型のテイスト（動画2・3）

`--style` で人型の描き方を選べます（動きは同じ）。`--style all` で5種類まとめて書き出します（`video3_combined_<テイスト>.mp4`）。

| `--style` | テイスト |
|---|---|
| `picto`（既定） | ピクトグラム |
| `mannequin` | デッサン人形（木製・関節の球・顔のない卵形の頭） |
| `plush` | ぬいぐるみ（太く丸い手足・くまの頭・縫い目） |
| `silhouette` | 人のシルエット（実際の体型に近い太さ・手と靴） |
| `real` | リアル寄りのイラスト（肌・髪・顔・パーティーの服装） |

`silhouette` と `real` の服装は `data/picks.txt` の各行に「名前,dress」と書くとワンピース、「名前,suit」か何も書かないとスーツになります。描き方は `figures.py` にまとまっています。

## 進行のパターン（動画2・3）

`--plan` で途中のポーズと進行を選べます（`--plan all` で全種類まとめて書き出し。`--style` と組み合わせ可）。

| `--plan` | 進行 |
|---|---|
| `standard`（既定） | 両手を上げる → それぞれポーズ → 伸び縮み → 逆関節 → 巻きつけ |
| `sync` | 全員で同じ振り（右・左・右・左・頭上で手を合わせる）→ 決めポーズ → そのまま一気に組む |
| `sync_pause` / `sync_huddle` / `sync_stretch` / `sync_bend` | `sync` の「そのまま組みます」の前にワンクッション（一拍おく / 3人で寄る / 体を伸ばす / 関節を逆に曲げる） |
| `gradual` | 両手を上げる → 腕が少し長く → 胴が少し長く → 膝が少し逆に → 肘が逆に → しなる → 巻きつけ |
| `onebyone` | 両手を上げる → 手をつなぐ → 1人ずつ順番に形になる（輪 → 青い葉 → 黄色い葉と茎） |

例: `python3 video3_combined.py --style silhouette --plan all` → `video3_combined_silhouette_<進行>.mp4`

## 席次表

- 席次表（.xlsx）は `data/seating.xlsx` に置きます。別の場所なら `--seating パス` で指定します。参列者の個人情報なので `data/` と、名前入りの書き出し動画（`*.mp4`）はリポジトリに入れません（`.gitignore` 済み）。
- 読み込みは `seating.py`。名前の右に「様」があるセルを名前、その2つ上を続柄として読み、卓は上にある卓名（A〜R）で決めます。会場図（`hall.py`）は、席次表の卓名の位置をそのまま会場の配置にしています。
- 動画2・3の3人は `data/picks.txt`（1行に1人、左から。「名前,dress」で服装も指定可。なければ A・C・B 卓の先頭の人）、動画1で最初にゆっくり紹介する人は `video1_formation.py` の `SLOW_TABLES`（各卓でフロアに一番近い人）で変えられます。

## 再生成

```bash
pip install pillow numpy imageio imageio-ffmpeg
pip install openpyxl
python3 video1_formation.py                   # → video1_formation.mp4
python3 video2_ginkgo.py                      # → video2_ginkgo.mp4
python3 video2_ginkgo.py --long               # → video2_ginkgo_long.mp4（長い版）
python3 video3_combined.py                    # → video3_combined.mp4
python3 video3_combined.py --style all        # → 人型のテイスト5種類
python3 video1_formation.py --tsukkomi        # 動画1の最後に文字のツッコミ演出も入れる場合
python3 video1_formation.py out.mp4 --preview # 確認用の静止画 PNG も書き出す
```

日本語フォントは IPA ゴシックを使っています（パスは `common.py` の `FONT_PATHS` で変更できます）。

## 東大のマークについて（動画 2）

- `assets/utokyo_mark.png`（マークのみ）と `assets/utokyo_logo.png`（下に「東京大学」の文字付き。右側に出すもの）は、いただいたマーク画像を `trace_mark.py` で色ごとに塗り直して高解像度にしたものです。人がなぞる形（`video2_ginkgo.py` の `pose_yellow` / `pose_blue` / `pose_ring`）も、この画像の座標に合わせてあります。
- 別の画像から作り直すときは `python3 trace_mark.py 画像.jpg 中心x 中心y 半径` を実行します。
- 東京大学のマークは大学の登録商標です。公開する範囲や使い方には気をつけてください（このリポジトリが公開リポジトリの場合、画像もそのまま公開されます）。
