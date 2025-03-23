#!/usr/bin/env python3
import torch
from torch import nn
from torchvision import models
from visualizer import consts


class FCNResNet101(nn.Module):
    """House a FCN_ResNet101 model from pytorch, adjusted for Mekides' interface."""

    def __init__(self, categories):
        super().__init__()
        self.model = models.segmentation.fcn_resnet101(
            weights=models.segmentation.fcn.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        )
        self._categories = nn.ParameterDict(
            {i: nn.Parameter(torch.Tensor(0)) for i in categories}
        )
        num_categories = len(self._categories)
        self.model.classifier[4] = nn.Conv2d(512, num_categories, 1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_categories, 1)

    @property
    def categories(self):
        return self._categories

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def load(self, trained_file):
        """Load trained model from file to local dictionary."""
        # Create model and load data to memory
        # @Wilhelmsen: Alter to include more models, when we include more models
        # if it's possible to do it in the same function...
        checkpoint = torch.load(
            trained_file, map_location=consts.DEVICE, weights_only=False
        )

        # Make necessary alterations to state_dict before loading into model
        # @Wilhelmsen: This can surely be foreshortened, perhaps with list comprehension...?
        state_dict = checkpoint["state_dict"]
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_key = key.removeprefix("module.")
            new_state_dict[new_key] = value

        checkpoint["state_dict"] = new_state_dict
        self.load_state_dict(checkpoint["state_dict"], strict=True)

        self.to(consts.DEVICE)
        self.eval()
