# -*- coding : UTF-8 -*-
# @file   : AttentionUnet3D.py
# @Time   : 2024-05-21 16:32
# @Author : wmz
import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        attention=False,
        feat_channels=[16, 32, 64, 128, 256, 512],
    ):
        super().__init__()
        # Encoder parts
        self.enc1 = Block(in_channels, feat_channels[0], attention)
        self.enc2 = Block(feat_channels[0], feat_channels[1], attention)
        self.enc3 = Block(feat_channels[1], feat_channels[2], attention)
        self.enc4 = Block(feat_channels[2], feat_channels[3], attention)
        # self.enc5 = Block(feat_channels[3], feat_channels[4], attention)
        # Decoder parts
        # self.dec4 = Block(feat_channels[4], feat_channels[3], attention)
        self.dec3 = Block(feat_channels[3], feat_channels[2], attention)
        self.dec2 = Block(feat_channels[2], feat_channels[1], attention)
        self.dec1 = Block(feat_channels[1], feat_channels[0], attention)
        # Upsamplers
        self.up4 = UpConv(feat_channels[4], feat_channels[3])
        self.up3 = UpConv(feat_channels[3], feat_channels[2])
        self.up2 = UpConv(feat_channels[2], feat_channels[1])
        self.up1 = UpConv(feat_channels[1], feat_channels[0])
        # Downsamplers
        self.down = nn.MaxPool3d(2, 2)
        # Output Conv
        self.out_conv = nn.Conv3d(
            feat_channels[0], num_classes, 1, 1, bias=False)

    def forward(self, x):
        e1 = self.enc1(x)  # 192x192x192
        e2 = self.enc2(self.down(e1))  # 96x96x96
        e3 = self.enc3(self.down(e2))  # 48x48x48
        e4 = self.enc4(self.down(e3))  # 24x24x24
        # e5 = self.enc5(self.down(e4))  # 12x12x12
        # d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg = self.out_conv(d1)  # 分割牙齿前景
        return seg


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, attention: bool = False):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.scSE = scSE(out_channels) if attention else None

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        if self.scSE:
            return self.scSE(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up(x)


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channels, 1, 1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.conv1x1(U)
        q = self.norm(q)
        return U * q  # 广播机制


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_squeeze = nn.Conv3d(
            in_channels, in_channels//2, 1, bias=False)
        self.conv_excitation = nn.Conv3d(
            in_channels//2, in_channels, 1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.pool(U)
        z = self.conv_squeeze(z)
        z = self.conv_excitation(z)
        z = self.norm(z)
        return U * z.expand_as(U)


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sSE = sSE(in_channels)
        self.cSE = cSE(in_channels)

    def forward(self, U):
        U_sSE = self.sSE(U)
        U_cSE = self.cSE(U)
        return U_sSE + U_cSE


def export_onnx(model, input, input_names, output_names, modelname):
    model.eval()
    dummy_input = input
    torch.onnx.export(model, dummy_input, modelname,
                      export_params=True,
                      verbose=False,
                      opset_version=12,
                      input_names=input_names,
                      output_names=output_names, dynamic_axes={'input': {2: "z", 3: "y", 4: "x"}, 'output': {2: "z", 3: "y", 4: "x"}})


if __name__ == "__main__":
    device = "cpu"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, num_classes=1, attention=True).to(device)
    x = torch.rand(1, 1, 224, 144, 144)
    x = x.to(device)
    output = model(x)
    print(output.shape)
    with torch.no_grad():
        input = torch.randn(1, 1, 224, 144, 144, device=device)
        input_names = ['input']
        output_names = ['output']
        export_onnx(model, input, input_names, output_names, "test.onnx")
