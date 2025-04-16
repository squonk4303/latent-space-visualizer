#!/usr/bin/env python3
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


def dataset_to_tensors(image_paths: list):
    """
    Take a list of image files and return them as converted to tensors.

    Returned tensors are of shape `height * width * RGB`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Open images for processing with PIL.Image.open
    dataset = [PIL.Image.open(image).convert("RGB") for image in image_paths]

    preprocessing = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(consts.STANDARD_IMG_SIZE),
            torchvision.transforms.ToTensor(),
        ]
    )

    tensors = [preprocessing(img).unsqueeze(0).to(device) for img in dataset]

    return tensors


def preliminary_dim_reduction_iii(model, layer, files):

    from pathlib import Path
    import plotly.express as px
    import random

    # Shuffle and assign unique colors
    base_colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    random.shuffle(base_colors)

    # Construct the full color map with hex and RGB
    color_map = {}
    for i, cat in enumerate(model.categories):
        hex_color = base_colors[i % len(base_colors)]
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[j : j + 2], 16) for j in (0, 2, 4))
        color_map[cat] = {"hex": "#" + hex_color, "rgb": rgb}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing = torchvision.transforms.Compose(
        [
            # @Wilhelmsen: NOTE: Image size is reduced for testing
            torchvision.transforms.Resize(consts.STANDARD_IMG_SIZE),
            torchvision.transforms.ToTensor(),
        ]
    )

    dominant_categories = []
    features = []
    filtered_filenames = []
    masks = []
    predicted_labels = []
    valid_paths = []

    # Register hook; hooked_feature is a list for its pointer-like qualities
    features_list = []  # @Wilhelmsen: change this to a dict or something; for elegance
    hook_location = getattr(model.model.backbone, layer)
    hook_handle = hook_location.register_forward_hook(hooker(features_list))

    # tqdm = lambda a, desc: a  # @Wilhelmsen: TEMP: Quick tqdm-disabler
    for image_location in tqdm(files[0:81], desc="processing imgs"):
        image = PIL.Image.open(image_location).convert("RGB")
        image = preprocessing(image)
        features_list.clear()

        with torch.no_grad():
            # Forward pass to get output and trigger hook
            output = model(image.unsqueeze(0).to(device))

        # --- Handle Hook ---
        feature_map = features_list[0]
        # Ensure array type and array element type
        feature_vector = torch.tensor(feature_map, dtype=torch.float32).to(device)
        # Apply GAP
        feature_vector = (
            torch.nn.functional.adaptive_avg_pool2d(feature_vector, (1, 1))
            .squeeze()
            .cpu()
            .numpy()
        )

        # --- Handle Output ---
        # Process prediction logits (assume output["out"] shape: [1, C, H, W])
        logits = output["out"]
        # Sigmoid transforms within the same dimensionality
        # Here gets shape (C, H, W)
        pred_mask = torch.sigmoid(logits).squeeze()
        # Set values under threshold to 0
        # @Wilhelmsen: this gate is disabled because the values in our set are all blocked by it
        pred_mask[pred_mask < 0.2] = 0
        print("*** pred_mask:", pred_mask)

        if pred_mask.ndim == 3 and pred_mask.shape[0] == len(model.categories):
            # Expected shape: (H, W)
            pred_class = torch.argmax(pred_mask, dim=0).cpu().numpy()

            # Identify background pixels
            background_mask = (pred_mask.sum(dim=0) == 0).cpu().numpy()
            pred_class[background_mask] = -1
        else:
            raise ValueError("Unexpected prediction shape.")

        # Determine dominant class, excluding background
        valid_classes = pred_class[pred_class != -1]
        if valid_classes.size > 0:
            unique, counts = np.unique(valid_classes, return_counts=True)
            dominant_class_idx = int(unique[np.argmax(counts)])
            dominant_category = model.categories[dominant_class_idx]
            dominant_categories.append(dominant_category)
            predicted_labels.append(dominant_class_idx)
            filtered_filenames.append(image_location)
            features.append(feature_vector)
            valid_paths.append(image_location)
        else:
            print(f"Skipping {image_location} due to no valid class predictions.")
            continue

        # Generate false-color mask using RGB values
        height, width = pred_class.shape
        false_color = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx, category in enumerate(model.categories):
            mask = pred_class == class_idx
            if category in color_map:
                false_color[mask] = color_map[category]["rgb"]

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

    hook_handle.remove()
    features = np.array(features).reshape(len(features), -1)
    return features, valid_paths, dominant_categories, masks


def preliminary_dim_reduction_2(model, layer, label, files):
    """
    Reduce the dimensionality of vectors in certain layer of nn-model

    to something t-SNE can more easily digest.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # @Wilhelmsen: Be parat for adding hooks in the arguments here.
    #       Possible implementaiton: dict of hooks as parameter;
    #       a for loop sets up hooks and which attributes they connect to
    #       then a dictionary is returned based on something to identify
    #       the hooks and a list of what they've hooked. Possibly
    #           Actually, maybe you can just plant the hook outside the function...

    # Register hook; hooked_feature is a list for its pointer-like qualities
    hooked_feature = []
    labels, paths, features = [], [], []

    # Keep in mind that what attribute to hook may be different per model type
    hook_handle = getattr(model, layer).register_forward_hook(hooker(hooked_feature))

    preprocessing = torchvision.transforms.Compose(
        [
            # @Wilhelmsen: NOTE: Image size is reduced for testing
            torchvision.transforms.Resize(28),
            torchvision.transforms.ToTensor(),
        ]
    )

    # @Wilhelmsen: NOTE input temporarily truncated /!/!\!\
    for path in tqdm(files[0:4], desc=f"Extracting from {label}"):
        hooked_feature.clear()
        # Load image as tensor
        try:
            image = PIL.Image.open(path).convert("RGB")
        except OSError:
            # @Wilhelmsen: Improve this error message; happens when binary is messed up in file
            # Such as when you open the bin in a text editor and remove a random segment
            # Not that I would know anything about that
            # @Wilhelmsen: Also find the error message at empty file
            # @Wilhelmsen: Also find the error message at non-binary file
            # (not that being non-binary is an error, just if you're a png)
            print(f"Truncated file read; continuing without image in {path}")
            continue

        if image is None:
            print(f"Couldn't convert {path}")
            # features.append(path, None)
            continue

        # Apply preprocessing on image and send to device
        image = preprocessing(image).unsqueeze(0).to(device)

        # Pass model through image; this triggers the hook
        # Calling the model takes quite a bit of time, it does
        with torch.no_grad():
            _ = model(image)
            feature_map = hooked_feature[0]

        # Transform hooked feature to a PyTorch tensor
        if not isinstance(feature_map, torch.Tensor):
            feature_map = torch.tensor(feature_map, dtype=torch.float32, device=device)
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

        labels.append(label)
        paths.append(path)
        features.append(feature_vector)

    features = np.asarray(features).reshape(len(features), -1)

    hook_handle.remove()

    return paths, features


def preliminary_dim_reduction(model, image_tensors, layer):
    """Reduce the dimensionality of tensors to something t-SNE can more easily digest."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        for img in tqdm(image_tensors):
            hooked_feature.clear()
            _ = model(img)  # Forward the model to let the hook do its thang

            feature_map = hooked_feature[0]

            # Ensure hooked feature is a PyTorch tensor
            if not isinstance(feature_map, torch.Tensor):
                feature_map = torch.tensor(
                    feature_map, dtype=torch.float32, device=device
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
    """Reduce features' dimensionality by t-SNE and return the 2d/3d coordinates."""
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

    Keep in mind that assigning an existing list to a variable actually
    provides a reference to the list, as opposed to a copy of which.
    """

    def f(module, args, output):
        t.append(output.detach().cpu().numpy())
        # print("From hook, latest append:", t[-1].shape)

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
    load_location = open_dialog.for_some_file(parent=parent)

    if load_location:
        with open(load_location, "rb") as f:
            data_object = pickle.load(f)

        return data_object
