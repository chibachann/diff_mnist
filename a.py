
import torch
import torch.nn as nn
import torch.optim as optim
from model_utils import UNet
from data_utils import prepare_data
from noise_utils import noise_process
import matplotlib.pyplot as plt
import os

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの準備
train_loader = prepare_data(batch_size=256)

# モデルのインスタンス化
model = UNet(n_channels=1, n_classes=1).to(device)

# 損失関数と最適化アルゴリズムの定義
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ノイズスケジュールの定義
beta_start = 0.0001
beta_end = 0.02
num_timesteps = 1000
beta_schedule = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

# ディレクトリが存在しない場合は作成
os.makedirs('saved_models', exist_ok=True)

# 学習ループ
num_epochs = 40
for epoch in range(num_epochs):
    for batch_idx, (clean_images, _) in enumerate(train_loader):
        clean_images = clean_images.to(device)
        batch_size = clean_images.shape[0]

        # ノイズの付与
        t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
        noisy_images, noise = noise_process(clean_images, t, beta_schedule)

        # モデルの出力
        predicted_noise = model(noisy_images)

        # 損失の計算
        loss = criterion(predicted_noise, noise)

        # 勾配の計算と更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 進捗の表示
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # モデルの保存
        torch.save(model.state_dict(), os.path.join('saved_models', f'model_epoch_{epoch+1}.pth'))

