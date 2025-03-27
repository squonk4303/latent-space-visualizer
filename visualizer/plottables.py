#!/usr/bin/env python3
import dataclasses
import numpy as np
import torch


@dataclasses.dataclass
class Plottables:
    model: torch.nn.Module = None
    selected_layer: str = None
    dataset_intermediary: list[torch.tensor] = None
    dataset_plottable: list[torch.tensor] = None
    image_plottable: tuple = None

    def __eq__(self, other):
        """
        This is because torch and numpy are so cagey about whether two of
        their arrays are equal. Like come on.
        """
        are_true = []

        for a, b in zip(dataclasses.astuple(self), dataclasses.astuple(other)):

            if type(a) is not type(b):
                are_true.append(False)

            else:
                match type(a):
                    case torch.Tensor:
                        are_true.append(torch.equal(a, b))

                    case np.ndarray:
                        are_true.append(np.array_equal(a, b))

                    case _:
                        are_true.append(a == b)

        return all(are_true)

    def __repr__(self):
        d = dataclasses.asdict(self)
        return "".join([f"{k}:    {v}\n" for k, v in d.items()])
