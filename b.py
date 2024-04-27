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
train_loader = prepare_data(batch_size=128)

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


# 評価関数
@torch.no_grad()
def evaluate(model, num_images=5):
    model.eval()
    clean_images, _ = next(iter(train_loader))
    clean_images = clean_images[:num_images].to(device)
    
    # ノイズの付与
    t = torch.randint(0, num_timesteps, (num_images,), device=device).long()
    noisy_images, _ = noise_process(clean_images, t, beta_schedule)

    # デノイズの過程
    denoised_images = noisy_images.clone()
    for i in range(num_timesteps - 1, -1, -1):
        t_tensor = torch.tensor([i] * num_images, device=device).long()
        predicted_noise = model(denoised_images)
        alpha = 1 - beta_schedule[t_tensor]
        alpha_bar = torch.cumprod(alpha, dim=0)[-1]

        # print(f"denoised_images: {denoised_images.shape}")
        a = ((1 - alpha) / torch.sqrt(1 - alpha_bar))
        a = a.view(a.size(0), 1, 1, 1)
        # print(f"((1 - alpha) / torch.sqrt(1 - alpha_bar)) : {a.shape}")
        # print(f"predicted_noise: {predicted_noise.shape}")
        # print(f"torch.sqrt(alpha): {torch.sqrt(alpha).shape}")
        b = torch.sqrt(alpha)
        b = b.view(b.size(0), 1, 1, 1)        
        denoised_images = (denoised_images - a * predicted_noise) / b

    # 結果の表示
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    for i in range(num_images):
        axs[i, 0].imshow(clean_images[i].cpu().squeeze().numpy(), cmap='gray')
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(noisy_images[i].cpu().squeeze().numpy(), cmap='gray')
        axs[i, 1].set_title(f"Noisy Image (t={t[i].item()})")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(denoised_images[i].cpu().squeeze().numpy(), cmap='gray')
        axs[i, 2].set_title("Denoised Image")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# 学習済みモデルの読み込み
model.load_state_dict(torch.load("./saved_models/model_epoch_40.pth"))

# 評価の実行
evaluate(model)
