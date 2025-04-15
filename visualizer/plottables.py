#!/usr/bin/env python3
import dataclasses
import numpy as np
import torch


@dataclasses.dataclass
class Plottables:
    """A mutable tuple of related elements for plotting."""

    path: str
    label: str
    features: np.ndarray = None
    tsne: list = None


@dataclasses.dataclass
class SavableData:
    model: torch.nn.Module = None
    layer: str = ""
    paths: list[str] = dataclasses.field(default_factory=list)
    dataset_location: str = ""
    dataset_intermediary: torch.tensor = None

    # Map features and other relevant values to a label
    old_plottables: dict[str, list[Plottables]] = dataclasses.field(
        default_factory=dict
    )

    # Map related features, paths, labels, &c together
    plottables: list[Plottables] = dataclasses.field(default_factory=list)

    labels: list = dataclasses.field(default_factory=list)
    masks: list = dataclasses.field(default_factory=list)
    paths: list = dataclasses.field(default_factory=list)
    two_dee: list = dataclasses.field(default_factory=list)

    def __eq__(self, other):
        """
        Handle attributes with exceptional equalities with care.

        This is just because torch and numpy are so cagey about whether two
        of their arrays are equal. Like come on; if you use a comparison
        operator, you want a boolean value in return. Get real.
        """
        for a, b in zip(dataclasses.astuple(self), dataclasses.astuple(other)):
            if type(a) is not type(b):
                return False
            else:
                if hasattr(a, "state_dict"):
                    # If it's a nn.Module, don't bother comparing
                    return True
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
