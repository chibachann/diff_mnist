import torch

def noise_process(x_0, t, beta_schedule):

    # ノイズの蓄積量を計算
    alpha_bar_t = torch.cumprod(1 - beta_schedule, dim=0)[t]


    # ガウス雑音をサンプリング
    epsilon = torch.randn_like(x_0)


    alpha_bar_t =  alpha_bar_t.view(alpha_bar_t.size(0), 1, 1, 1)

    # ノイズを付与
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon



    return x_t, epsilon