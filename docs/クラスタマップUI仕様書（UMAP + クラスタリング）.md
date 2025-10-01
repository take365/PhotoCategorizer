# クラスタマップUI仕様書（UMAP + クラスタリング）

## 目的

* 画像コレクションを **UMAP** により2次元に投影し、**クラスタリング(DBSCAN/HDBSCAN)** によって群として可視化する。
* 群を背景色で区切ることで、ユーザーが画像群の全体構造を直感的に把握できるようにする。
* 代表画像や属性要約を表示し、群単位での探索や検索に接続できるようにする。
* **UIの基本構成は意味軸マップと合わせる**（入力エリア／マップエリア／詳細ペイン／メニュー）。

## 概要

* モード切替タブを設け、「ロケーション」「被写体」「画像」の3種類のクラスタマップを表示可能とする。

  * **ロケーション**: 属性 location テキストを埋め込み→UMAP。
  * **被写体**: 属性 subject テキストを埋め込み→UMAP。
  * **画像**: OpenCLIP画像埋め込み→UMAP。
* 各モードはUMAP座標・クラスタIDを事前計算・保存して利用。
* クラスタは背景色の塗り分け（凹包またはKDE等高線）により区切って表示する。
* ノイズ点（クラスタ外）は別色で表示する。

## 画面要素（意味軸マップに準拠）

* **入力エリア**:

  * モード切替タブ: location / subject / image
  * 表示件数スライダー
  * メタ情報の更新ボタン、ズームリセットボタン、チャンク切り替えボタン。
* **散布図エリア**:

  * 極小サムネイルを点で表示。
  * 背景をクラスタごとに半透明色で塗り分け（凹包ポリゴン）。
  * ホバー: ツールチップにクラスタIDと属性概要を表示。
  * クリック: 詳細ペインに群情報を表示。ダブルクリックで個別画像のオーバーレイを開く。
* **詳細ペイン**:

  * 群の代表画像3–5枚（重心近傍・多様性考慮）。
  * 件数、全体比率、密度指標。
  * 属性要約（location/subject/tone/style/composition の上位値、および keywords）。
  * クイックアクション:

    * 「この群で検索」: キーワードを組み合わせたクエリを属性検索に遷移。
    * 「意味軸マップで表示」: 代表語を使って scatter 画面を新規タブで開く。

## バックエンド仕様

* **UMAP**: metric=cosine, n_neighbors≈30, min_dist≈0.15, random_state固定。
* **クラスタリング**: DBSCAN（min_samples=10、epsは k-距離法で自動推定）。
* **クラスタ境界生成**: 簡易凹包（monotonic chain + 補正）で GeoJSON ポリゴンを作成。
* **保存形式**:

  * coords.npy (N×2座標)
  * labels.npy (クラスタID or -1 for noise)
  * hulls.geojson (境界ポリゴン)

## API例

### メタ情報

```
GET /clusters/{mode}/meta
→ { total, cluster_count, noise_ratio, clusters, hulls, params, bounds }
```

### 座標チャンク取得

```
GET /clusters/{mode}/chunks/{chunk}?size=200
→ { chunk, total_chunks, points: [{image_id, x, y, cluster_id, thumbnail, attributes, color}, ...] }
```

### クラスタ詳細

```
GET /clusters/{mode}/detail/{cluster_id}
→ { count, ratio, centroid, density, attribute_summary, representative_images: [...] }
```

### ジャンプ

```

※ 意味軸マップ連携はクライアント側で scatter 画面のクエリパラメータを生成して遷移する。
```

## メニュー

* 意味軸マップと同様のメニュー構造に「クラスタマップ」を追加し、両画面を行き来できるようにする。
