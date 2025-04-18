#!/usr/bin/env python3
import torch
import random
from torch import nn
from torchvision import models
from visualizer import consts


class _SegmentationInterface(nn.Module):

    def __init__(self):
        super().__init__()

    @property
    def categories(self):
        return list(self._categories.keys())

    @categories.setter
    def categories(self, value):
        if value is not None:
            self._categories = nn.ParameterDict(
                {i: nn.Parameter(torch.Tensor(0)) for i in value}
            )
        else:
            self._categories = nn.ParameterDict()


class FCNResNet101(_SegmentationInterface):
    """House a FCN_ResNet101 model from pytorch, adjusted for Mekides' interface."""

    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet101(
            weights=models.segmentation.fcn.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        )
        self.categories = None
        self.colormap = {}

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def load(self, trained_file):
        """Load a trained model from path into current object."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model data from checkpoint in file
        checkpoint = torch.load(trained_file, map_location=device, weights_only=False)

        # Strip "module." from any key-names in state_dict
        checkpoint["state_dict"] = {
            key.removeprefix("module."): value
            for key, value in checkpoint["state_dict"].items()
        }

        # Get all expected categories and apply them
        # NOTE that this only sets categories that were
        # originally prefixed with "_categogies."
        categories = [
            i.removeprefix("_categories.")
            for i in checkpoint["state_dict"]
            if i.startswith("_categories.")
        ]
        self.categories = categories

        # Make a dict which maps all categories to a unique color
        self.colormap = {
            label: color
            for label, color in zip(
                self.categories,
                random.sample(consts.COLORS32, k=len(self.categories)),
            )
        }

        # Set certain submodules based on amount of categories
        num_categories = len(self._categories)
        self.model.classifier[4] = nn.Conv2d(512, num_categories, 1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_categories, 1)

        self.load_state_dict(checkpoint["state_dict"], strict=True)
        self.to(device)
        self.eval()


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # Contracting path (Encoder)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)

        # Expanding path (Decoder)
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final Convolution Layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)  # 64 channels
        enc2 = self.encoder2(torch.nn.functional.max_pool2d(enc1, 2))  # 128 channels
        enc3 = self.encoder3(torch.nn.functional.max_pool2d(enc2, 2))  # 256 channels
        enc4 = self.encoder4(torch.nn.functional.max_pool2d(enc3, 2))  # 512 channels
        enc5 = self.encoder5(torch.nn.functional.max_pool2d(enc4, 2))  # 1024 channels

        # Decoder
        dec4 = self.upconv4(enc5)  # Up-sample
        dec4 = torch.cat((enc4, dec4), dim=1)  # Skip connection
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)
