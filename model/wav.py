class WavLayer(nn.Module):
    def __init__(self, level, h, w):
        super(WavLayer, self).__init__()
        self.level = level
        self.high_freq_weight = nn.Parameter(torch.randn(h, w) * 0.02)
        self.low_freq_weight = nn.Parameter(torch.randn(h // (2 ** level), w // (2 ** level)) * 0.02)
        self.conv = nn.Conv2d(2, 16, kernel_size=3, padding=1) 

    def forward(self, x):
        x = x[:, 0]  # take the first channel
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

        # Upsample low-frequency component to high-frequency size
        low_freq_output = low_freq_output.unsqueeze(1)  # [B, 1, 28, 28]
        low_freq_output = torch.nn.functional.interpolate(low_freq_output, size=(h, w), mode='bilinear', align_corners=False)  # [B, 1, 224, 224]
        high_freq_output = high_freq_output.unsqueeze(1)  # [B, 1, 224, 224]

        # Concatenate high-frequency and low-frequency components
        output = torch.cat([high_freq_output, low_freq_output], dim=1)  # [B, 2, 224, 224]
        output = self.conv(output)  # [B, 16, 224, 224]
        return output
