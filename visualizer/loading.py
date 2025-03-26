#!/usr/bin/env python3
from sklearn.manifold import TSNE
from tqdm import tqdm
import mmap
import numpy as np
import pickle
import PIL
import tempfile
import torch
import torchvision

from visualizer import consts
from visualizer.plottables import Plottables


def dataset_to_tensors(image_paths: list):
    """
    Take a list of image files and return them as converted to tensors.

    Returned tensors are of shape `height * width * RGB`.
    """
    # Open images for processing with PIL.Image.open
    dataset = [PIL.Image.open(image).convert("RGB") for image in image_paths]

    preprocessing = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(consts.STANDARD_IMG_SIZE),
            torchvision.transforms.ToTensor(),
        ]
    )

    tensors = [preprocessing(img).unsqueeze(0).to(consts.DEVICE) for img in dataset]

    return tensors


def preliminary_dim_reduction(model, image_tensors, layer):
    """Reduce the dimensionality of tensors to something t-SNE can more easily digest."""
    # Register hook and yadda yadda
    # Otherwise use function find_layer to let user choose layer
    # Then use gitattr() to dynamically select the layer based on user choice...!
    hooked_feature = []
    features = []
    # A list so    ~~^^ that we can use it as pointer and with isinstance
    # Plant the hook   in   layer4  ~~~~vvv
    hook_handle = model.model.backbone.layer4.register_forward_hook(
        hooker(hooked_feature)
    )

    # Preliminary dim. reduction per tensor
    with torch.no_grad():
        for img in image_tensors:
            hooked_feature.clear()
            _ = model(img)  # Forward the model to let the hook do its thang

            feature_map = hooked_feature[0]

            # Ensure feature map is a PyTorch tensor
            # @Wilhelmsen: find out what on earth the point of this is.
            # Do it when in the encapsulation process
            # And see whether hooked_feature can be something besides a list
            if not isinstance(feature_map, torch.Tensor):
                feature_map = torch.tensor(
                    feature_map, dtype=torch.float32, device=consts.DEVICE
                )
            # Reduce dimensionality using Global Average Pooling (GAP)
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_avg_pool2d.html#torch.nn.functional.adaptive_avg_pool2d
            # @Wilhelmsen: Opiton for different dim.reduction techniques.
            # Do it when in the encapsulation process
            feature_vector = (
                torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
                .squeeze()
                .cpu()
                .numpy()
            )
            features.append(feature_vector)

    # Ensure features have correct 2D shape; (num_samples, num_features)
    # @Wilhelmsen: Just find out what the point is. Do it in encapsulation process.
    features = np.array(features).reshape(len(features), -1)

    # Remove hook
    hook_handle.remove()

    return features


def apply_tsne(features, target_dimensions=2):
    """Applies t-SNE to the features and returns the result."""
    # Ensure a reasonable/legal perplexity value
    perplexity_value = min(30, len(features) - 1)

    tsne_conf = TSNE(
        n_components=target_dimensions,
        perplexity=perplexity_value,
        random_state=consts.seed,
    )

    reduced_features = tsne_conf.fit_transform(features)

    return reduced_features


def hooker(t: list):
    """
    Return a hook function which appends model output to the given list.

    Keep in mind that assigning an existing list to a variable
    actually provides a "pointer" to the list
    """

    def f(module, args, output):
        t.append(output.detach().cpu().numpy())
        print("From hook, latest append:", t[-1].shape)

    return f


def layer_summary(loaded_model, start_layer=0, end_layer=0):
    """
    Summarises selected layers from a given model objet.
    If endlayer is left blank only return one layer.
    If start layer is left blank returns all layers.
    If both layers are specified returns from startlayer up to
    and including the endlayer!
    """
    # Sets basic logic and variables
    all_layers = False
    if not end_layer:
        end_layer = start_layer
    if not start_layer:
        all_layers = True

    input_txt = str(loaded_model)
    target = "layer"
    # Assigns targetlayers for use in search later
    next_layer = target + str(end_layer + 1)
    target += str(start_layer)

    """
    At some point in this function an extraction function is to be added
    to filter the information and only return the useful information and attributes
    to be added to the list. For now it takes the entire line of information.
    """

    # Create a temporary data file to store data in a list
    lines = []
    with tempfile.TemporaryFile("wb+", 0) as file:
        file.write(input_txt.encode("utf-8"))
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        while True:
            byteline = mm.readline()
            if byteline:
                lines.append(byteline.decode("utf-8"))
            else:
                break
        mm.close()

    # Returns selected layers
    found = False
    eol = False
    new = 0
    for i, line in enumerate(lines):
        if all_layers:
            pass
        elif target in line:
            found = True
        elif next_layer in line:
            eol = True
            new = i
        if all_layers or found and not eol:
            print(f"{i}: {line}", end="")

    # End of print
    if all_layers:
        print("\nEOF: no more lines")
    else:
        print(f"\nNext line is {new}: {lines[new]}")
