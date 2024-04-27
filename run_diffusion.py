import torch
import matplotlib.pyplot as plt
from data_utils import prepare_data
from noise_utils import noise_process

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの準備
train_loader = prepare_data(batch_size=32)

# ノイズスケジュールの定義
beta_start = 0.0001
beta_end = 0.02
num_timesteps = 1000
beta_schedule = torch.linspace(beta_start, beta_end, num_timesteps).to(device)

# データの取得
data, _ = next(iter(train_loader))
data = data.to(device)

# t=1~29のタイムステップでノイズを付与
timesteps = range(1, 1000)
noisy_images = []
for t in timesteps:
    noisy_data, _ = noise_process(data, t, beta_schedule)
    noisy_images.append(noisy_data[0])


# 表示する画像のインデックスを選択
display_indices = [0, 50, 100, 200, 500, 999]

# 元の画像とノイズが付与された画像を表示
fig, axs = plt.subplots(1, 6, figsize=(15, 3))
axs = axs.flatten()

axs[0].imshow(data[0].cpu().squeeze().numpy(), cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

for i, idx in enumerate(display_indices[1:], start=1):
    axs[i].imshow(noisy_images[idx-1].cpu().squeeze().numpy(), cmap='gray')
    axs[i].set_title(f"Noisy Image (t={idx})")
    axs[i].axis('off')

plt.tight_layout()
plt.show()