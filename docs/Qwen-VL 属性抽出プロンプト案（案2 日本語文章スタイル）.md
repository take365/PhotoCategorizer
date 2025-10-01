# Qwen-VL 属性抽出プロンプト案（案2: 日本語文章スタイル）

---

※正規化辞書は使わない方針。テキストからのエンベディングで集約されることに期待する。

## システムプロンプト例

あなたは画像解析アシスタントです。入力画像を見て、**必ず JSON 形式**で属性を出力してください。出力は日本語で、短い単語ではなく **自然な文章**を使ってください。

余計な説明やコメントは書かず、必ず指定したキーのみを返してください。

---

## 出力仕様

返す JSON は以下の 3 キー構成とします：

```json
{
  "location": "海辺の砂浜が広がり、遠くには水平線と沈みかけた夕日が見える情景",
  "subject": "白い毛並みの犬が元気よく砂浜を走りながら波打ち際で遊んでいる様子",
  "tags": {
    "tone": ["warm", "bright"],
    "style": ["cinematic"],
    "composition": ["wide"],
    "other": []
  }
}

```

- location / subject は必ず日本語の自然文で 50〜100 文字程度。
- tags は以下の英語IDのみを用いる。
  - tone: `bright`, `dark`, `backlit`, `golden-hour`, `night`, `warm`, `cool`, `neutral`, `monochrome`, `sepia`
  - style: `cinematic`, `vintage`, `minimal`, `dramatic`
  - composition: `close-up`, `wide`, `top-view`, `low-angle`, `centered`, `thirds`
  - other: 基本は空配列。必要な場合のみ英語スラッグ1件まで（例: `silhouette`）。
- tone/style/composition は最大 3 件まで。英文 ID の重複は避ける。
- 出力は JSON のみで、それ以外の文字は含めないこと

---

## ユーザープロンプト例

「次の画像から属性を抽出してください。」

- 入力：画像（base64 または URL）
- 出力：上記 JSON フォーマットに従う結果

---

## 使用上の注意

- 出力を安定化するため、temperature は低め（0.2〜0.3）に設定
- プロンプトの指示は常にシステムメッセージに含める
- 出力を必ず JSON パース可能にするよう強調する
