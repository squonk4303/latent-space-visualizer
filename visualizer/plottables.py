#!/usr/bin/env python3
import dataclasses
import numpy as np
import torch


@dataclasses.dataclass
class Plottables:
    model: torch.nn.Module = None
    selected_layer: str = None
    dataset_intermediary: torch.tensor = None
    dataset_plottable: torch.tensor = None
    image_plottable: tuple = None
    # TODO: What about dict with settings? Like whether user displayed in 2d or 3d and such

    def __eq__(self, other):
        """
        This is just because torch and numpy are so cagey about whether two
        of their arrays are equal. Like come on.
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
