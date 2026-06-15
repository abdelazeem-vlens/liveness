# -*- coding: utf-8 -*-
"""
MultiFTNet (refactored)
-----------------------
Keeps the core idea of the original Silent-Face-Anti-Spoofing repo: a main
classification head plus an auxiliary branch that regresses the Fourier
(magnitude) spectrum of the input, used only during training to shape the
shared backbone features.

Changes vs. the original:
  * Backbone is torchvision MobileNetV3 (large/small) instead of MiniFASNet.
  * Designed for larger inputs (e.g. 224x224) rather than 80x80.
  * 2-class (real/spoof) by default.
  * The FT branch taps the deepest backbone stage whose spatial size is still
    >= `tap_min_spatial`, and the FT target spatial size is taken from that
    tap (same rule the original used: target size == tap feature-map size).
"""

import torch
from torch import nn
from torchvision import models


# ---------------------------------------------------------------------------
# Auxiliary Fourier-spectrum generator (spatial size preserved by padding=1)
# ---------------------------------------------------------------------------
class FTGenerator(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(FTGenerator, self).__init__()
        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ft(x)


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------
def _build_backbone(name, pretrained):
    """Return a torchvision MobileNetV3 backbone (weights optional)."""
    name = name.lower()
    if name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
    else:
        raise ValueError(
            "Unsupported backbone '{}'. Use 'mobilenet_v3_large' or "
            "'mobilenet_v3_small'.".format(name))
    return net


def _resolve_tap_index(features, input_size, min_spatial, img_channel=3):
    """
    Probe the backbone's `features` and return the index of the DEEPEST block
    whose output spatial size is still >= `min_spatial`. The auxiliary branch
    taps the feature map right after that block.
    """
    x = torch.zeros(1, img_channel, input_size[0], input_size[1])
    tap = 0
    with torch.no_grad():
        for i, module in enumerate(features):
            x = module(x)
            if x.shape[-1] >= min_spatial:
                tap = i
    return tap


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class MultiFTNet(nn.Module):
    def __init__(self,
                 backbone="mobilenet_v3_large",
                 pretrained=True,
                 num_classes=2,
                 input_size=(224, 224),
                 tap_min_spatial=14,
                 img_channel=3):
        super(MultiFTNet, self).__init__()
        self.num_classes = num_classes
        self.img_channel = img_channel

        base = _build_backbone(backbone, pretrained)

        # Replace the final classifier layer with our num_classes head.
        in_feat = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_feat, num_classes)

        # Split the feature extractor at the FT tap point.
        features = list(base.features)
        tap_index = _resolve_tap_index(
            base.features, input_size, tap_min_spatial, img_channel)
        self.features_low = nn.Sequential(*features[:tap_index + 1])
        self.features_high = nn.Sequential(*features[tap_index + 1:])
        self.avgpool = base.avgpool
        self.classifier = base.classifier

        # Probe the tap feature map to size the FT branch and its target.
        with torch.no_grad():
            dummy = torch.zeros(1, img_channel, input_size[0], input_size[1])
            low = self.features_low(dummy)
        self.tap_channels = low.shape[1]
        self.ft_spatial = (low.shape[2], low.shape[3])  # (H, W)

        self.FTGenerator = FTGenerator(in_channels=self.tap_channels, out_channels=1)

        # Initialise ONLY the newly added modules (keep pretrained backbone).
        self._initialize_new_weights()

    def _initialize_new_weights(self):
        for m in self.FTGenerator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # The replaced classifier Linear keeps PyTorch's default init.

    def forward(self, x):
        low = self.features_low(x)          # tap feature map
        high = self.features_high(low)
        pooled = self.avgpool(high)
        pooled = torch.flatten(pooled, 1)
        cls = self.classifier(pooled)

        if self.training:
            ft = self.FTGenerator(low)
            return cls, ft
        return cls
