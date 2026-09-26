# フラッシュモブ「無理やん！」動画

縦型ショート動画（1080×1920 / 30fps / 効果音付き）を 2 本、Python で生成します。

| ファイル | 長さ | 内容 |
|---|---|---|
| `video1_formation.mp4` | 41 秒 | ◯（＝人）を 1 人ずつ紹介しながら配置 → V 字（ここまでは OK）→ 逆回転リング・8 の字・スピログラフ・すり抜け・ハート・星・全員 1 点に重なる・散開シャッフル → 3 倍速でもう 1 回 → 全員の動線がスパゲッティ状態に → **無理やん！** |
| `video2_ginkgo.mp4` | 17 秒 | デッサン人形 3 体が「東大のイチョウマーク作ります」→ 腕が 3 倍に伸びる → 関節が逆に曲がって「ボキッ」→ 首が 2 回転しながら巻きついてイチョウの葉 2 枚＋茎に → 「完成！」「※関節は考慮していません」→ **無理やん！** |

## 再生成

```bash
pip install pillow numpy imageio imageio-ffmpeg
python3 video1_formation.py                 # → video1_formation.mp4
python3 video2_ginkgo.py                    # → video2_ginkgo.mp4
python3 video1_formation.py out.mp4 --preview   # 確認用の静止画 PNG も書き出す
```

日本語フォントは IPA ゴシックを使っています（パスは `common.py` の `FONT_PATHS` で変更できます）。

### よくいじる場所
- テロップ・ボケの文言: `video1_formation.py` の `SEGMENTS` と `intros`、`video2_ginkgo.py` の `build()` 内の各シーン
- 人数・名前: `NAMES`（`N` も合わせて変更）
- 各パートの秒数: `SEGMENTS` と各 `add(秒数, ...)` の数値
- ツッコミの文字: `common.tsukkomi(img, u, word="無理やん！")`

ツッコミの声は入れていません。効果音「ドン」の位置（動画 1 は 38.4 秒、動画 2 は 14.4 秒）に、自分の声か音声合成の「無理やん！」を重ねるのがおすすめです。

## 注意
- イチョウは東京大学の公式ロゴをそのまま再現したものではなく、2 枚のイチョウの葉を組み合わせた「それっぽい」シルエットです。公開するときは、大学名やロゴの扱いに気をつけてください。

## 動画生成 AI（Sora / Veo / Kling など）で作る場合のプロンプト案

動画生成 AI は「◯の数や位置を正確に保つ」「決まった形を作る」のが苦手です。そのため、**フォーメーション部分は上のスクリプトで作ったものを使い、生成 AI には実写風のツッコミやオチのカットだけを任せる**やり方がおすすめです。

**動画 1（ツッコミのカット）**
> A Japanese office meeting room. A choreographer points at a whiteboard covered in chaotic colored spiral lines connecting 16 circles. Cut to a young man in a hoodie who slams both hands on the table and shouts in disbelief, comedic manga-style reaction, speed lines, handheld camera, 9:16 vertical.

**動画 2（人形がイチョウの形になるカット）**
> Three wooden artist mannequins standing on a table, stop-motion style. Their arms stretch like rubber, joints bend backwards impossibly, heads spin twice, and together they twist into the shape of two ginkgo leaves with a stem. Warm studio light, playful, 9:16 vertical, then a sudden freeze frame.

最後に字幕「無理やん！」を動画編集アプリ（CapCut など）で入れます。
