"""为旌平台超分模型（IR_256 ×4, first_dim=8）— 对应 E89a 权重。

来源：训练服务器 `/home/wuzhongqi/image_enhancement/models/vs_sr_model_nearest_cat_pixelshuffle_single_4.py`
E89a 配置：`e89a_ir256_strim_lpips_dim8.yaml`

模型配置：
1. 整体架构为 UNet+ResNet，UNet 用于学习残差。
2. UNet 采用步长为 2 的卷积实现下采样，采用转置卷积实现上采样，采用 ReLU6 激活函数。
3. UNet 采用 cat 实现跨层连接。
4. UNet 末端采用 PixelShuffle 实现上采样（为旌平台支持 PixelShuffle 算子）。
5. 模型先将原图上采样放大（最近邻插值），再加上 UNet 输出的残差，得到最终输出。

输入：单通道灰度 [N,1,H,W] float32，H/W 需能被 16 整除（256×192 ✓）
输出：单通道 [N,1,H*scale,W*scale] float32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IE_Net(nn.Module):
    def __init__(self, scale_factor=4, first_dim=8):
        super(IE_Net, self).__init__()
        dims = [first_dim * (2 ** i) for i in range(5)]
        self.scale_factor = scale_factor

        self.conv_first = nn.Sequential(
            nn.Conv2d(1, dims[0], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 2, 2, 0, bias=False),
            nn.Conv2d(dims[0], dims[1], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], 2, 2, 0, bias=False),
            nn.Conv2d(dims[1], dims[2], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], 2, 2, 0, bias=False),
            nn.Conv2d(dims[2], dims[3], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(dims[3], dims[3], 2, 2, 0, bias=False),
            nn.Conv2d(dims[3], dims[4], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.upsample1 = nn.ConvTranspose2d(dims[4], dims[3], 2, 2, 0, bias=False)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(dims[4], dims[3], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.upsample2 = nn.ConvTranspose2d(dims[3], dims[2], 2, 2, 0, bias=False)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(dims[3], dims[2], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.upsample3 = nn.ConvTranspose2d(dims[2], dims[1], 2, 2, 0, bias=False)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.upsample4 = nn.ConvTranspose2d(dims[1], dims[0], 2, 2, 0, bias=False)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], 3, 1, 1),
            nn.ReLU6(inplace=True)
        )
        self.conv_last = nn.Conv2d(dims[0], scale_factor ** 2, 3, 1, 1)
        self.depth_to_space = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        conv_first = self.conv_first(x)
        encoder1 = self.encoder1(conv_first)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        upsample1 = self.upsample1(encoder4)
        decoder1 = self.decoder1(torch.cat([upsample1, encoder3], 1))
        upsample2 = self.upsample2(decoder1)
        decoder2 = self.decoder2(torch.cat([upsample2, encoder2], 1))
        upsample3 = self.upsample3(decoder2)
        decoder3 = self.decoder3(torch.cat([upsample3, encoder1], 1))
        upsample4 = self.upsample4(decoder3)
        decoder4 = self.decoder4(torch.cat([upsample4, conv_first], 1))
        conv_last = self.conv_last(decoder4)
        out = self.depth_to_space(conv_last) + x_up
        return out


if __name__ == '__main__':
    scale_factor = 4
    first_dim = 8
    H, W = 192, 256
    model = IE_Net(scale_factor=scale_factor, first_dim=first_dim)
    x = torch.randn(1, 1, H, W)
    with torch.no_grad():
        out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f'Input:  {tuple(x.shape)}')
    print(f'Output: {tuple(out.shape)}')
    print(f'Params: {params / 1e6:.3f}M')
