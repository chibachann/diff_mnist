import torch
import torch.nn as nn
from model_utils import UNet
from data_utils import prepare_data
import matplotlib.pyplot as plt
import os

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのインスタンス化
model = UNet(n_channels=1, n_classes=1).to(device)

# ノイズスケジュールの定義
beta_start = 0.0001
beta_end = 0.02
num_timesteps = 1000
beta_schedule = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

# 画像生成関数
@torch.no_grad()
def generate_images(model, num_images=5):
    model.eval()
    
    # 完全なノイズの生成
    noise = torch.randn(num_images, 1, 28, 28).to(device)
    
    # 画像生成の過程
    generated_images = noise.clone()
    for i in range(num_timesteps - 1, -1, -1):
        t_tensor = torch.tensor([i] * num_images, device=device).long()
        predicted_noise = model(generated_images)
        alpha = 1 - beta_schedule[i]
        alpha_bar = torch.cumprod(alpha, dim=0)[-1]
        
        a = ((1 - alpha) / torch.sqrt(1 - alpha_bar))
        a = a.view(-1, 1, 1, 1)  # ブロードキャストするためにサイズを調整
        
        b = torch.sqrt(alpha)
        b = b.view(-1, 1, 1, 1)  # ブロードキャストするためにサイズを調整
        
        generated_images = (generated_images - a * predicted_noise) / b
        
        if i > 0:
            z = torch.randn_like(generated_images)
            sigma = torch.sqrt(beta_schedule[i])
            generated_images += sigma * z
    
    # 結果の表示
    fig, axs = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    for i in range(num_images):
        axs[i].imshow(generated_images[i].cpu().squeeze().numpy(), cmap='gray')
        axs[i].set_title(f"Generated Image {i+1}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# 学習済みモデルの読み込み
model.load_state_dict(torch.load("./saved_models/model_epoch_40.pth"))

# 画像生成の実行
generate_images(model)