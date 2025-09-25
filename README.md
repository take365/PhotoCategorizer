# PhotoCategorizer

ゼロショット画像分類で `studio/room/city/forest/sea/desert` と `color/mono` を自動推定する CLI ツールです。

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Classify images

```bash
photocat classify ./images --out-csv outputs/pred.csv --out-json outputs/pred.json
```

- `--move-to-class-dirs` を付けると `outputs/classified/{label}/{color}` にファイルを移動します。
- `--config` で別の YAML 設定ファイルを指定できます。

## Evaluate on labelled set

1. `eval_images/` に評価用画像を配置します。
2. `labels.csv` を用意し、`filename,label` 形式で保存します。
3. 次を実行します。

```bash
photocat eval ./eval_images --labels labels.csv --report outputs/report.md
```

Markdown のレポートに混同行列と自動確定率が出力されます。

- `--model both` を指定すると SigLIP と OpenCLIP を同時に評価し、結果は `outputs/pred.json` / `outputs/gallery.html` に保存されます。
- GPT による目視代替判定を追加したい場合は `.env` に `OPENAI_API_KEY` と `BATCH_MODEL` を設定し、`--mode gpt-off`（OpenCLIPのみ）/`--mode gpt-review`（閾値未満のみGPT）/`--mode gptall`（GPT優先）を選択してください。
- 現在の出力は `outputs/<timestamp>/` 以下にまとめられ、画像ファイルは `color/` / `grayscale/` → `カテゴリ名/` の順に移動されます（HTML も相対パスで閲覧可）。
- 最終割り振り一覧は `outputs/<timestamp>/summary.csv` と `report.md` に保存されます（CSV には `filename,path,final_label,color_mode` を出力）。
- 大規模運用で GPT のレート制御が必要な場合は `.env` に `GPT_TPM_LIMIT`（トークン/分）や `GPT_RPM_LIMIT`（リクエスト/分）を設定すると CLI 側で自動的にスロットルします。
- `.env` では `EVAL_MODE` / `EVAL_MODEL` / `GPT_BATCH_SIZE` / `GPT_RPS` など実行時のデフォルト値も指定できます（CLI で明示した引数が優先されます）。

### .env で設定できる主な項目

- `PIXABAY_API_KEY`, `OPENAI_API_KEY`, `BATCH_MODEL`：外部 API の認証情報と利用モデル名。
- `EVAL_MODE`：`gpt-off` / `gpt-review` / `gptall` から既定の GPT モードを選択。
- `EVAL_MODEL`：`openclip` / `both` / `siglip` のいずれかを既定モデルとして使用。
- `GPT_BATCH_SIZE`, `GPT_RPS`, `GPT_MAX_RETRIES`, `GPT_BACKOFF_BASE`：GPT 呼び出しのバッチサイズや送信レート、リトライ回数、バックオフ初期値を制御。
- `GPT_TPM_LIMIT`, `GPT_RPM_LIMIT`：組織のレートリミットに合わせた追加ガード。未設定時は CLI オプション値のみで管理。
- 他にも CLI オプションは `.env` の値で事前設定でき、環境依存の調整を簡素化できます。

### start.bat での一括実行

- Windows 環境では `start.bat` を実行すると `venv_win` の作成・依存関係のインストール・`photocat eval input` の実行まで自動化されます。
- `.env` に設定した `EVAL_MODE` や GPT 関連のデフォルト値がそのまま適用されるため、バッチを再実行するだけで最新の設定で評価できます。

### 対応ファイル形式

- 入力として扱える拡張子は `.jpg`, `.jpeg`, `.png`, `.webp` の4種類です。
- `.bmp` は GPT 推論が利用できないためスキップ対象になります（入力フォルダに残り、`report.md` に警告として記録されます）。

### 色判定の調整

- `config.yaml` の `color_judge.threshold` は飽和度平均の閾値（初期値 0.08）です。全体が淡い色調でも平均が閾値を超えれば color と判定されます。
- `color_judge.pixel_threshold` は「色付きとみなすピクセル」の飽和度閾値（初期値 0.12）で、`min_color_ratio` はその割合が一定以上（初期値 0.02 = 2%）なら色付きとみなす補助判定です。
- 淡色背景にアクセントカラーがある写真で mono 判定される場合は `min_color_ratio` を下げる、色付きノイズで誤検出する場合は `pixel_threshold` や `min_color_ratio` を上げると調整できます。

### カテゴリ定義の変更

- `config.yaml` の `classes` セクションでクラスとプロンプトを定義しています。
  ```yaml
  classes:
    city:
      - "a street scene in a city with buildings and roads"
      - "urban street with traffic and pedestrians"
  ```
- クラス名を追加・削除したい場合は、この辞書に項目を追記・削除してください。
- それぞれのクラスには少なくとも1つのプロンプト文を登録します。`photocat eval` / `photocat classify` 実行時はこの設定ファイルが読み込まれるため、変更後は再実行するだけで新しいカテゴリが反映されます。
- 別の設定を使いたい場合は `--config PATH/to/config.yaml` を指定して読み込んでください。

## Download sample assets from Pixabay

`.env` に `PIXABAY_API_KEY=...` を記載するか、環境変数で指定した上で次を実行します。

```bash
photocat pixabay-download --query "room interior" --preset background --limit 10 --out-dir images/pixabay_samples
```

`--preset` で `background` / `icon` / `item` を切り替えられ、パラメータ確認のみなら `--dry-run` を付けてください。

### Windows での対話型ダウンロード

- `pixabay_fetch.bat` を実行すると、キーワード（日本語入力可）と取得枚数を尋ねるダイアログが表示されます。
- バッチ内では自動的に `venv_win` を用意し、`photocat pixabay-download` を呼び出して `pixabay/<キーワード>/` 以下に画像と `pixabay_metadata.json` を保存します。
- 保存先フォルダ名の禁止文字は安全な `_` に置換されるため、Windows でもそのまま利用できます。
