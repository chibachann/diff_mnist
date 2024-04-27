# UNetアーキテクチャの詳細

## 入力
- チャネル数: `n_channels`（ここでは1）
- 次元数: `(batch_size, n_channels, height, width)`

## DoubleConv
`DoubleConv`は、UNetの基本的な畳み込みブロックです。以下の層から構成されています。

1. 畳み込み層（Conv2d）:
  - カーネルサイズ: 3
  - パディング: 1
  - 活性化関数: ReLU
  - バッチ正規化（BatchNorm2d）

2. 畳み込み層（Conv2d）:
  - カーネルサイズ: 3
  - パディング: 1
  - 活性化関数: ReLU
  - バッチ正規化（BatchNorm2d）

`DoubleConv`ブロックは、入力のチャネル数と出力のチャネル数を指定して初期化されます。中間のチャネル数は、出力のチャネル数と同じか、指定されない場合は出力のチャネル数と同じになります。

## Down
`Down`ブロックは、エンコーダ部分でダウンサンプリングを行うために使用されます。以下の層から構成されています。

1. マックスプーリング（MaxPool2d）:
  - カーネルサイズ: 2
  - ストライド: 2

2. DoubleConvブロック:
  - 入力のチャネル数と出力のチャネル数を指定して初期化

`Down`ブロックは、入力のチャネル数と出力のチャネル数を指定して初期化されます。マックスプーリング層によって空間次元が半分になり、DoubleConvブロックによって特徴マップが抽出されます。

## Up
`Up`ブロックは、デコーダ部分でアップサンプリングを行うために使用されます。以下の層から構成されています。

1. アップサンプリング:
  - バイリニア補間（Upsample）:
    - スケールファクター: 2
    - モード: 'bilinear'
    - アライン済みの角（align_corners）: True
  - または、トランスポーズ畳み込み（ConvTranspose2d）:
    - カーネルサイズ: 2
    - ストライド: 2

2. DoubleConvブロック:
  - 入力のチャネル数と出力のチャネル数を指定して初期化

`Up`ブロックは、入力のチャネル数と出力のチャネル数、およびアップサンプリング方法（バイリニア補間またはトランスポーズ畳み込み）を指定して初期化されます。アップサンプリング層によって空間次元が2倍になり、DoubleConvブロックによって特徴マップが抽出されます。

## OutConv
`OutConv`は、UNetの出力層です。以下の層から構成されています。

1. 畳み込み層（Conv2d）:
  - カーネルサイズ: 1
  - 活性化関数: なし

`OutConv`ブロックは、入力のチャネル数と出力のチャネル数を指定して初期化されます。出力のチャネル数は、通常、クラス数に対応します。

## エンコーダ
1. 入力 → DoubleConv (inc):
  - 入力チャネル数: `n_channels`
  - 出力チャネル数: 64
  - 次元数: `(batch_size, 64, height, width)`

2. DoubleConv (inc) → Down (down1):
  - 入力チャネル数: 64
  - 出力チャネル数: 128
  - 次元数: `(batch_size, 128, height/2, width/2)`

3. Down (down1) → Down (down2):
  - 入力チャネル数: 128
  - 出力チャネル数: 256
  - 次元数: `(batch_size, 256, height/4, width/4)`

4. Down (down2) → Down (down3):
  - 入力チャネル数: 256
  - 出力チャネル数: 512
  - 次元数: `(batch_size, 512, height/8, width/8)`

5. Down (down3) → Down (down4):
  - 入力チャネル数: 512
  - 出力チャネル数: 1024 // factor（bilinear interpolationを使用する場合は512）
  - 次元数: `(batch_size, 1024 // factor, height/16, width/16)`

## デコーダ
6. Down (down4) → Up (up1):
  - 入力チャネル数: 1024
  - 出力チャネル数: 512 // factor
  - 次元数: `(batch_size, 512 // factor, height/8, width/8)`
  - スキップ接続: Down (down3)の出力をUp (up1)の出力とチャネル方向に連結

7. Up (up1) → Up (up2):
  - 入力チャネル数: 512
  - 出力チャネル数: 256 // factor
  - 次元数: `(batch_size, 256 // factor, height/4, width/4)`
  - スキップ接続: Down (down2)の出力をUp (up2)の出力とチャネル方向に連結

8. Up (up2) → Up (up3):
  - 入力チャネル数: 256
  - 出力チャネル数: 128 // factor
  - 次元数: `(batch_size, 128 // factor, height/2, width/2)`
  - スキップ接続: Down (down1)の出力をUp (up3)の出力とチャネル方向に連結

9. Up (up3) → Up (up4):
  - 入力チャネル数: 128
  - 出力チャネル数: 64
  - 次元数: `(batch_size, 64, height, width)`
  - スキップ接続: DoubleConv (inc)の出力をUp (up4)の出力とチャネル方向に連結

## 出力
10. Up (up4) → OutConv (outc):
   - 入力チャネル数: 64
   - 出力チャネル数: `n_classes`（ここでは1）
   - 次元数: `(batch_size, n_classes, height, width)`

## スキップ接続
- エンコーダの各レベルの出力は、対応するデコーダのレベルの入力とチャネル方向に連結されます。
- これにより、エンコーダで抽出された特徴マップの空間情報がデコーダに直接伝達され、詳細な情報が復元されます。

UNetは、エンコーダでダウンサンプリングを行いながら特徴を抽出し、デコーダでアップサンプリングとスキップ接続を利用して元の解像度を復元します。このアーキテクチャにより、画像のセグメンテーションや生成タスクにおいて高い性能を発揮します。