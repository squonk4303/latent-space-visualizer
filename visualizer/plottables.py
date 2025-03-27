#!/usr/bin/env python3
import dataclasses

# NOTE: Importing torch increases plottables.py individual test run-time by ~2.69 times
# NOTE: I'm not dealing with that


@dataclasses.dataclass
class Plottables:
    model: object = None  # Actually type is nn.Moule
    selected_layer: str = None
    dataset_intermediary: list = None  # Type is really list[torch.Tensor]
    dataset_plottable: list = None  # Type is really list[torch.Tensor]
    image_plottable: tuple = None

    def __repr__(self):
        d = dataclasses.asdict(self)
        return "".join([f"{k}:    {v}\n" for k, v in d.items()])
