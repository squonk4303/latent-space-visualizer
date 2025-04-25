#!/usr/bin/env python3
from collections.abc import Callable
from pathlib import Path
import os
import pickle

from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import PIL
import torch
import torchvision

from visualizer import consts, open_dialog
from visualizer.plottables import SavableData


# Try to use cuML for GPU acceleration (optional)
use_gpu_tsne = False
try:
    from cuml.manifold import TSNE as cuTSNE

    use_gpu_tsne = True
except ImportError:
    pass


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
    # 30 is usually the default
    perplexity_value = min(4, len(features) - 1)

    if use_gpu_tsne:
        tsne_conf = cuTSNE(
            n_components=target_dimensions,
            perplexity=perplexity_value,
            random_state=consts.seed,
        )
    else:
        tsne_conf = TSNE(
            n_components=target_dimensions,
            perplexity=perplexity_value,
            n_jobs=-1,
            random_state=consts.seed,
        )

    reduced_features = tsne_conf.fit_transform(features)
    return reduced_features


def pca():
    not_implemented_yet(pca)


def umap():
    not_implemented_yet(umap)


def trimap():
    not_implemented_yet(trimap)


def pacmap():
    not_implemented_yet(pacmap)


def not_implemented_yet(f: Callable):
    raise NotImplementedError(
        f'Dimensionality-reduction function "{f.__name__}" not implemented yet!'
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
    load_location = open_dialog.for_some_file(
        parent=parent, caption=consts.LOAD_FILE_DIALOG_CAPTION
    )

    if load_location:
        with open(load_location, "rb") as f:
            data_object = pickle.load(f)

        return data_object
