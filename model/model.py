"""
ResNet50 + CBAM (Convolutional Block Attention Module) for COVID-19 Radiography
4-class classification: COVID-19, Normal, Lung Opacity, Viral Pneumonia
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ChannelAttention(nn.Module):
    """通道注意力模块 — 关注"什么"特征是有意义的"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力模块 — 关注"哪里"是有意义的区域"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out


class CBAM(nn.Module):
    """CBAM: 通道注意力 + 空间注意力串联"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResNet50CBAM(nn.Module):
    """ResNet50 + CBAM 分类模型"""

    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.4):
        super(ResNet50CBAM, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.cbam = CBAM(in_channels=2048, reduction_ratio=16, kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate - 0.1),
            nn.Linear(256, num_classes),
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(num_classes=4, pretrained=True, device=None):
    """创建模型的便捷函数"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNet50CBAM(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)
