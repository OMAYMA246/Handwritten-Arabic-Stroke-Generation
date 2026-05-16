import torch
import torch.nn as nn
from torchvision import models

class ArabicLetterEncoder(nn.Module):
    """
    Architecture exacte du notebook de votre binôme
    MobileNetV2 + tête de projection 128D + normalisation L2
    """
    def __init__(self, embedding_dim=128):
        super(ArabicLetterEncoder, self).__init__()

        # MobileNetV2 — pas de poids ImageNet (on charge les siens)
        self.backbone = models.mobilenet_v2(weights=None)

        # Modifier première couche : RGB → Grayscale (1 canal)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Classifier final — exactement comme votre binôme
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        # Normalisation L2 — exactement comme votre binôme
        return nn.functional.normalize(x, p=2, dim=1)