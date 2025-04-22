#!/usr/bin/env python3
from collections.abc import Callable
from pathlib import Path
import mmap
import os
import pickle
import tempfile

from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import PIL
import torch
import torchvision

from visualizer import consts, open_dialog
from visualizer.plottables import SavableData


def preliminary_dim_reduction_iii(model, layer, files, progress):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_to_fit = 128 if consts.flags["truncate"] else consts.STANDARD_IMG_SIZE
    preprocessing = torchvision.transforms.Compose(
        (
            torchvision.transforms.Resize(size_to_fit),
            torchvision.transforms.ToTensor(),
        )
    )

    dominant_categories = []
    features = []
    masks = []
    valid_paths = []

    # Register hook; hooked_feature is a list for its pointer-like qualities
    features_list = []  # @Wilhelmsen: change this to a dict or something; for elegance
    hook_location = getattr(model.model.backbone, layer)
    hook_handle = hook_location.register_forward_hook(hooker(features_list))

    files = files[:12] if consts.flags["truncate"] else files
    progress.setMaximum(len(files))
    progress.set_visible(True)
    progress.reset()
    for image_location in tqdm(files, desc="processing imgs"):
        progress()
        image = PIL.Image.open(image_location).convert("RGB")
        image = preprocessing(image).unsqueeze(0).to(device)
        features_list.clear()

        # Forward pass to get output and trip hook
        with torch.no_grad():
            output = model(image)

        # --- Handle Data From Hook ---
        # Convert feature to tensor of type float32
        # And apply GAP
        feature_vector = torch.tensor(features_list[0], dtype=torch.float32).to(device)
        feature_vector = (
            torch.nn.functional.adaptive_avg_pool2d(feature_vector, (1, 1))
            .squeeze()
            .cpu()
            .numpy()
        )

        # --- Handle Model Output ---
        # Process prediction logits (assume output["out"] list of logits, shape: [1, C, H, W])
        # Here gets shape (C, H, W)
        logits = output["out"]
        pred_mask = torch.sigmoid(logits).squeeze()
        # Set values under threshold to 0
        # @Wilhelmsen: Make the threshold based on a factor of the average
        pred_mask[pred_mask < 0.2] = 0

        # Identify background pixels, and denote with -1
        if pred_mask.ndim == 3 and pred_mask.shape[0] == len(model.categories):
            # Expected shape: (H, W) which is a binary mask
            pred_class = torch.argmax(pred_mask, dim=0).cpu().numpy()
            background_mask = (pred_mask.sum(dim=0) == 0).cpu().numpy()
            pred_class[background_mask] = -1
        else:
            raise ValueError("Unexpected prediction shape.")

        # Determine dominant class; ignore background
        valid_classes = pred_class[pred_class != -1]
        if valid_classes.size > 0:
            unique, counts = np.unique(valid_classes, return_counts=True)
            dominant_class_idx = int(unique[np.argmax(counts)])
            dominant_category = model.categories[dominant_class_idx]
            dominant_categories.append(dominant_category)
            features.append(feature_vector)
            valid_paths.append(image_location)
        else:
            print(f"Skipping {image_location} due to no valid class predictions.")
            progress.skipped_image()
            continue

        # Generate false-color mask using RGB values
        height, width = pred_class.shape
        false_color = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx, category in enumerate(model.categories):
            mask = pred_class == class_idx
            if category in model.colormap:
                false_color[mask] = PIL.ImageColor.getcolor(
                    model.colormap[category], "RGB"
                )

        # Set background pixels to black
        false_color[pred_class == -1] = (0, 0, 0)

        # Save false-color mask
        false_color_img = PIL.Image.fromarray(false_color)
        mask_dir = "tsne_visualizations/mask"
        os.makedirs(mask_dir, exist_ok=True)
        filename = Path(image_location)
        filename = filename.stem
        mask_path = os.path.join(mask_dir, f"{filename}_mask.png")
        false_color_img.save(mask_path)
        masks.append(false_color_img)
        # print(f"Saved false-color segmentation mask: {mask_path}")

    progress()
    progress.set_visible(False)
    hook_handle.remove()
    features = np.array(features).reshape(len(features), -1)
    return features, valid_paths, dominant_categories, masks


def tsne(features, target_dimensions=2):
    """Reduce features' dimensionality by t-SNE and return the 2d/3d coordinates."""
    # Ensure a reasonable/legal perplexity value
    perplexity_value = min(30, len(features) - 1)

    # @Wilhelmsen: Include option to use mutlithreaded tsne
    tsne_conf = TSNE(
        n_components=target_dimensions,
        perplexity=perplexity_value,
        random_state=consts.seed,
    )

    reduced_features = tsne_conf.fit_transform(features)
    return reduced_features


def pca():
    undefined_dim_reduction_technique(pca)


def umap():
    undefined_dim_reduction_technique(umap)


def trimap():
    undefined_dim_reduction_technique(trimap)


def pacmap():
    undefined_dim_reduction_technique(pacmap)


def undefined_dim_reduction_technique(f: Callable):
    raise NotImplementedError(
        f"Dim. reduction function {f} not implemented yet!"
    )


def hooker(t: list):
    """
    Return a hook function which appends model output to the given list.

    Keep in mind that assigning an existing list to a variable actually
    provides a reference to the list, as opposed to a copy of which.
    """

    def f(module, args, output):
        t.append(output.detach().cpu().numpy())

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


def quickload(load_location=consts.QUICKSAVE_PATH):
    """Load python object from pickle file."""
    with open(load_location, "rb") as f:
        data_obj = pickle.load(f)

    return data_obj


def quicksave(data_obj: SavableData, save_location=consts.QUICKSAVE_PATH):
    """Save python object to pickle file."""
    # If parent directory doesn't exist, create it (including its progenitors)
    parent_dir = os.path.abspath(os.path.join(save_location, os.pardir))
    os.makedirs(parent_dir, exist_ok=True)

    with open(save_location, "wb") as f:
        pickle.dump(data_obj, f)

    print(f"Saved to {save_location}")


def save_to_user_selected_file(data_obj: SavableData, parent):
    """
    Open a dialog to select from where to load a data object.

    For use in actions and buttons.
    """
    save_location, _ = open_dialog.to_save_file(parent=parent)

    if save_location:
        with open(save_location, "wb") as f:
            pickle.dump(data_obj, f)

        print(f"Saved to {save_location}")
        return save_location
    else:
        return False


def load_by_dialog(parent) -> object:
    """
    Open a dialog to select from where to load a data object.

    For use in actions and buttons.
    """
    load_location = open_dialog.for_some_file(parent=parent, caption=consts.LOAD_FILE_DIALOG_CAPTION)

    if load_location:
        with open(load_location, "rb") as f:
            data_object = pickle.load(f)

        return data_object
