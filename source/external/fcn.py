#!/usr/bin/env python3
import torch
from torch import nn
from torchvision import models


class FCNResNet101(nn.Module):
    """
    House a FCN_ResNet101 model from pytorch, adjusted for Mekides' interface.
    """
    def __init__(self, categories):
        super().__init__()
        self.model = models.segmentation.fcn_resnet101(
            weights=models.segmentation.fcn.
                FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        )
        self._categories = nn.ParameterDict(
            {i: nn.Parameter(torch.Tensor(0)) for i in categories})
        num_categories = len(self._categories)

        self.model.classifier[4]     = nn.Conv2d(512, num_categories, 1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_categories, 1)

    @property
    def categories(self):
        return self._categories

    def forward(self, image: torch.Tensor):
        return self.model(image)
