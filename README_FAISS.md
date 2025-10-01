# マルチ属性検索 PoC 簡易ガイド

1. **属性抽出 (Qwen のみ)**
   ```bash
   venv/bin/python -m photocat.cli attr-poc data/val2017 \
       --limit 100 \
       --use-qwen --no-use-gemma
   ```
   - `.env` に `LMSTUDIO_BASE_URL=http://172.25.192.1:1234/v1` と
     `LMSTUDIO_QWEN_MODEL=qwen/qwen2.5-vl-7b` を入れておくと、
     コマンド引数なしで LM Studio の設定を拾います。
   - 出力は `outputs/attr_poc/attr_results.json` と `attr_report.html`。

2. **Faiss + SQLite 構築**
   ```bash
   venv/bin/python -m photocat.cli attr-index outputs/attr_poc/attr_results.json \
       --reset
   ```
   - `.env` に `LMSTUDIO_EMBED_MODEL=text-embedding-embeddinggemma-300m-qat`
     をセットすると埋め込みモデル指定も省略可。
   - 出力: `outputs/vector_index` 配下に `meta.db`, `attr_*.faiss`, `images.faiss`。

3. **Web UI を起動して検索する**
   ```bash
   venv/bin/python -m photocat.cli attr-serve --host 0.0.0.0 --port 8000
   ```
   - `.env` に LM Studio の URL と埋め込みモデルを設定済みなら追加引数不要。
   - ブラウザで `http://localhost:8000` にアクセス。上部フォームで
     キーワードや属性別テキスト／重み、画像アップロードを指定可能。
   - 各結果カードの「詳細を表示」内に、属性 JSON、類似度内訳バー、
     「画像類似」「ロケーション類似」などの再検索ボタンを配置。

4. **クラスタマップ用の事前計算を実行**
   ```bash
   venv/bin/python -m photocat.cli cluster-precompute --index-dir outputs/vector_index
   ```
   - location / subject / image の各モードで UMAP + DBSCAN を走らせ、
     `outputs/vector_index/clusters/<mode>/` に `coords.npy` などを生成。
   - `--mode location --mode subject` のようにモードを限定することも可能。
   - `--limit` や `--min-samples` などで前処理パラメータを調整できます。

5. **検索利用例**
   ```python
   from photocat.attr_index import AttributeIndexer, IndexPaths, create_text_client
   paths = IndexPaths(Path('outputs/vector_index'))
   client = create_text_client(None, None)  # .env に依存する
   ix = AttributeIndexer(paths, text_client=client, text_model='text-embedding-nomic-embed-text-v1.5')
   results = ix.search_attributes('location', '夕暮れの海辺', top_k=5)
   ```

6. **.env 設定の例**
   ```env
   LMSTUDIO_BASE_URL=http://172.25.192.1:1234/v1
   LMSTUDIO_API_KEY=lm-studio
   LMSTUDIO_QWEN_MODEL=qwen/qwen2.5-vl-7b
   LMSTUDIO_GEMMA_MODEL=google/gemma-3-12b
   LMSTUDIO_EMBED_MODEL=text-embedding-embeddinggemma-300m-qat
   ```

この設定で `attr-poc` → `attr-index` が引数ほぼ無しで回ります。
