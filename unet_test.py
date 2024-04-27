# train.py

import torch
from model_utils import UNet
from data_utils import prepare_data

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの準備
train_loader = prepare_data(batch_size=32)

# モデルのインスタンス化
model = UNet(n_channels=1, n_classes=1).to(device)

# モデルの動作確認
with torch.no_grad():
    # データローダーからサンプルを取得
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch.to(device)

    # モデルの出力を計算
    output = model(sample_batch)

    # 入力と出力のサイズを比較
    print("Input shape:", sample_batch.shape)
    print("Output shape:", output.shape)