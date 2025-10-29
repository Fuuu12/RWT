class WavLayer(nn.Module):
    def __init__(self, level, h, w):
        super(WavLayer, self).__init__()
        self.level = level
        self.high_freq_weight = nn.Parameter(torch.randn(h, w) * 0.02)
        self.low_freq_weight = nn.Parameter(torch.randn(h // (2 ** level), w // (2 ** level)) * 0.02)
        self.conv = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # 输入通道为2

    def forward(self, x):
        x = x[:, 0]  # 取第一个通道
        B, h, w = x.shape
        high_freq_output = torch.zeros_like(x)
        low_freq_output = torch.zeros(B, h // (2 ** self.level), w // (2 ** self.level), device=x.device)
        
        for i in range(B):
            data = x[i].cpu().numpy()
            coeffs = pywt.wavedec2(data, 'db2', mode='periodization', level=self.level)
            coeffs[0] /= np.abs(coeffs[0]).max()
            for detail_level in range(self.level):        
                coeffs[detail_level + 1] = [
                    d / np.abs(d).max() for d in coeffs[detail_level + 1]
                ]
            low_freq = torch.from_numpy(coeffs[0]).to(x.device)  # [28, 28]
            arr, _ = pywt.coeffs_to_array(coeffs)  # [224, 224]
            arr = torch.from_numpy(arr).to(x.device)

            high_freq_output[i] = arr * self.high_freq_weight
            low_freq_output[i] = low_freq * self.low_freq_weight

        # 上采样低频到高频尺寸
        low_freq_output = low_freq_output.unsqueeze(1)  # [B, 1, 28, 28]
        low_freq_output = torch.nn.functional.interpolate(low_freq_output, size=(h, w), mode='bilinear', align_corners=False)  # [B, 1, 224, 224]
        high_freq_output = high_freq_output.unsqueeze(1)  # [B, 1, 224, 224]

        # 拼接高频和低频
        output = torch.cat([high_freq_output, low_freq_output], dim=1)  # [B, 2, 224, 224]
        output = self.conv(output)  # [B, 16, 224, 224]
        return output

class WavCon(nn.Module):
    def __init__(self):
        super(WavCon, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class FusionModule(nn.Module):
    def __init__(self, embed_dim=768):
        super(FusionModule, self).__init__()
        self.wav_fc = nn.Linear(embed_dim, embed_dim)
        self.vit_fc = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_vit, x_wav):
        x_vit = self.vit_fc(x_vit)
        x_wav = self.wav_fc(x_wav)
        combined = torch.cat([x_vit, x_wav], dim=-1)
        weights = self.attn(combined)
        out = x_vit * weights[:, 0].unsqueeze(-1) + x_wav * weights[:, 1].unsqueeze(-1)
        return out
