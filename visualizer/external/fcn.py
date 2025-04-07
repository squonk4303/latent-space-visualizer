#!/usr/bin/env python3
import torch
from torch import nn
from torchvision import models
from visualizer import consts


class FCNResNet101(nn.Module):
    """House a FCN_ResNet101 model from pytorch, adjusted for Mekides' interface."""

    def __init__(self, categories=None):
        super().__init__()
        self.model = models.segmentation.fcn_resnet101(
            weights=models.segmentation.fcn.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        )
        self.categories = categories

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, categories):
        if categories is not None:
            self._categories = nn.ParameterDict(
                {i: nn.Parameter(torch.Tensor(0)) for i in categories}
            )
        else:
            self._categories = nn.ParameterDict()

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def load(self, trained_file):
        """Load a trained model from path into current object."""
        # Load model data from checkpoint in file
        checkpoint = torch.load(
            trained_file, map_location=consts.DEVICE, weights_only=False
        )

        # Strip "module." from any key-names in state_dict
        checkpoint["state_dict"] = {
            key.removeprefix("module."): value
            for key, value in checkpoint["state_dict"].items()
        }

        # Get all expected categories and apply them
        categories = [
            i.removeprefix("_categories.")
            for i in checkpoint["state_dict"]
            if i.startswith("_categories.")
        ]
        self.categories = categories

        # Set certain submodules based on the categories
        num_categories = len(self.categories)
        self.model.classifier[4] = nn.Conv2d(512, num_categories, 1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_categories, 1)

        self.load_state_dict(checkpoint["state_dict"], strict=True)
        self.to(consts.DEVICE)
        self.eval()
