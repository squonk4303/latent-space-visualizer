#!/usr/bin/env python3
import dataclasses
import numpy as np
import torch


@dataclasses.dataclass
class Features:
    """
    Class to hold relevant data related to features.

    Mostly defined to improve interface with class Plottables > >
    """

    label: str
    path: str
    feature: np.array
    # @Linnea: Be glad I didn't name these "toody" and "threedy"
    two_dee = None
    three_dee = None
    color = None


@dataclasses.dataclass
class Plottables:
    model: torch.nn.Module = None
    selected_layer: str = None
    dataset_intermediary: torch.tensor = None
    dataset_plottable: torch.tensor = None
    image_plottable: tuple = None
    # TODO: What about dict with settings? Like whether user displayed in 2d or 3d and such

    # Map features and relevant values to a label
    # Note that Feature must be initialized with label, path, and feature
    # features: Feature = Feature()
    labels: list = dataclasses.field(default_factory=list)
    paths: list = dataclasses.field(default_factory=list)
    features: list = dataclasses.field(default_factory=list)

    def __eq__(self, other):
        """
        This is just because torch and numpy are so cagey about whether two
        of their arrays are equal. Like come on; if you use a comparison
        operator, you want a boolean value in return. Get real.
        """
        for a, b in zip(dataclasses.astuple(self), dataclasses.astuple(other)):
            if type(a) is not type(b):
                return False
            else:
                match type(a):
                    case torch.Tensor:
                        if not torch.equal(a, b):
                            return False
                    case np.ndarray:
                        if not np.array_equal(a, b):
                            return False
                    case _:
                        if a != b:
                            return False
        return True

    def __repr__(self):
        d = dataclasses.asdict(self)
        return "".join([f"{k}:    {v}\n" for k, v in d.items()])
