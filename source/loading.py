#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import torchvision
import mmap
import tempfile
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QFileDialog,
)

import consts
import external.fcn as fcn


class FileDialogManager:
    """
    Class to simplify handling of files and file dialogs.

    Methods:
        find_file -- generic function to open a qfiledialog; for wrapping
        find_some_file -- wrapper for find_file which has filters set for all files
        find_picture_file -- wrapper for find_file which has filters set for pictures
        find_trained_model_file -- wrapper for find_file which has filters set for trained nn models
    """

    def __init__(self, parent_window):
        """Initilalizes with a parent window to say to whom any of its spawned window is a child."""
        self.parent = parent_window

    def find_file(
        self,
        file_filters=consts.FILE_FILTERS.values(),
        options=QFileDialog.Option.ReadOnly,
    ):
        """
        Launch a file dialog and return the filepath and selected filter.

        Note that the first element in file_filters is used as the initial file filter.
        """
        # Test whether file_filters is a subset of-- or equal to, the legal file filters
        if not set(file_filters) <= set(consts.FILE_FILTERS.values()):
            raise RuntimeError("Unacceptable list of file filters")

        # Generate Qt-readable filter specifications
        file_filters = list(file_filters)
        initial_filter = file_filters[0]
        filters = ";;".join(file_filters)

        # This function opens a nifty Qt-made file dialog
        filepath, selected_filter = QFileDialog.getOpenFileName(
            parent=self.parent,
            filter=filters,
            initialFilter=initial_filter,
            options=options,
        )

        return filepath, selected_filter

    def find_directory(self):
        """Launch a file dialog where user is prompted to pick out a directory."""
        # @Wilhelmsen: Make sure this actually lets the user CHOOSE a directory
        path, _ = self.find_file(
            options=QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.ReadOnly
        )
        return path

    def find_some_file(self):
        """Launch a file dialog where user is prompted to pick out any file their heart desires."""
        path, _ = self.find_file()
        return path

    def find_picture_file(self):
        """Launch file dialog where user is intended to pick out a graphical image file."""
        filters = [
            consts.FILE_FILTERS["pictures"],
            consts.FILE_FILTERS["whatever"],
        ]
        path, _ = self.find_file(filters)
        return path

    def find_trained_model_file(self):
        """Launch file dialog where user is intended to pick out the file for a trained nn-model."""
        filters = [
            consts.FILE_FILTERS["pytorch"],
            consts.FILE_FILTERS["whatever"],
        ]
        path, _ = self.find_file(filters)
        return path


class AutoencodeModel:  # @Wilhelmsen: methinks this can be renamed to "ModelManager"
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = []
        self.image_tensors = []
        self.models = {}          # @Wilhelmsen: Perhaps there should be one model per object
        # @Wilhelmsen make the transformed image size a const, maybe choosable
        self.preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(640),
                torchvision.transforms.ToTensor(),
            ]
        )

    def load_model(self, name, trained_file, categories):
        """Load trained model from file to local dictionary."""
        # Create model and load data to memory
        # @Wilhelmsen: Alter to include more models, when we include more models
        # if it's possible to do it in the same function...
        model_obj = fcn.FCNResNet101(categories)
        checkpoint = torch.load(
            trained_file, map_location=self.device, weights_only=False
        )

        # Make necessary alterations to state_dict before loading into model
        state_dict = checkpoint["state_dict"]
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_key = key.removeprefix("module.")
            new_state_dict[new_key] = value

        checkpoint["state_dict"] = new_state_dict
        model_obj.load_state_dict(checkpoint["state_dict"], strict=True)

        model_obj.to(self.device)
        model_obj.eval()

        self.models[name] = model_obj
        return name

    def single_image_to_tensor(self, image_path) -> torch.tensor:
        """Convert image in path to tensor we can use."""
        single_image = Image.open(image_path).convert("RGB")
        # @Wilhelmsen: How long does this take? Do benchmarking.
        tensor = self.preprocessing(single_image).unsqueeze(0).to(self.device)
        return tensor

    def dataset_to_tensors(self, dataset):
        """
        TODO: Uses the file dialog to locate a dir (or zip file maybe) in
        which to scan for valid images and load them into memory. Also make
        an effort to state how many images are loaded, because *that* has a
        lot of implications that are relevant to the user.
        """
        # @Wilhelmsen: Going to have to make it so this function instead
        # takes either the path to the directory, a list of strings for the
        # filepaths, or an entire dam list of all the PIL.Images. I think I
        # would prefer it take the stringpath to the directory, then
        # extract the PIL.Images itself. More memory efficient, methonks.
        # TEMP: Hard-coded right now
        # @Wilhelmsen: Image.open opens the file and it remains open until the data is processed
        dataset = [
            Image.open(consts.GRAPHICAL_IMAGE).convert("RGB"),
            Image.open("pics/animals10/gallina/1000.jpeg").convert("RGB"),
            Image.open("pics/animals10/gallina/1001.jpeg").convert("RGB"),
            Image.open("pics/animals10/gallina/100.jpeg").convert("RGB"),
            Image.open("pics/animals10/gallina/1010.jpeg").convert("RGB"),
            Image.open("pics/animals10/gallina/1013.jpeg").convert("RGB"),
        ]

        # Save it as an attribute, because we want to append-to and reuse these
        self.image_tensors = [
            self.preprocessing(img).unsqueeze(0).to(self.device) for img in dataset
        ]

    def the_whole_enchilada(self, model_name):
        """
        Does a lot of things; to be encapsulated
        Verily based on example code we got from Mekides.
        """
        model_obj = self.models[model_name]

        # Register hook and yadda yadda

        # Plant the hook       in   layer4  ~~~~vvv
        # Otherwise use function find_layer to let user choose layer
        # Then use gitattr() to dynamically select the layer based on user choice...!
        hooked_feature = []
        # A list so    ~~^^ that we can use it as pointer and with isinstance later
        hook_handle = model_obj.model.backbone.layer4.register_forward_hook(
            hooker(hooked_feature)
        )

        # Find data set using this function:
        # dataset = FileDialogManager.find_dir/zip()
        data_paths = None
        self.dataset_to_tensors(data_paths)

        # Preliminary dim. reduction per tensor
        with torch.no_grad():
            for img in self.image_tensors:
                hooked_feature.clear()
                _ = model_obj(img)  # Forward the model to let the hook do its thang

                feature_map = hooked_feature[0]

                # Ensure feature map is a PyTorch tensor
                # @Wilhelmsen: find out what on earth the point of this is.
                # Do it when in the encapsulation process
                if not isinstance(feature_map, torch.Tensor):
                    feature_map = torch.tensor(
                        feature_map, dtype=torch.float32, device=self.device
                    )

                # Reduce dimensionality using Global Average Pooling (GAP)
                # https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_avg_pool2d.html#torch.nn.functional.adaptive_avg_pool2d
                # @Wilhelmsen: Opiton for different dim.reduction techniques.
                # Do it when in the encapsulation process
                feature_vector = (
                    F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze().cpu().numpy()
                )
                self.features.append(feature_vector)

        # Ensure features have correct 2D shape; (num_samples, num_features)
        # @Wilhelmsen: Just find out what the point is. Do it in encapsulation process.
        self.features = np.array(self.features).reshape(len(self.features), -1)

        # Ensure a reasonable/legal perplexity value
        perplexity_value = min(30, len(features) - 1)

        # Then finally apply t-SNE
        tsne = TSNE(
            n_components=2, perplexity=perplexity_value, random_state=const.SEED
        )
        reduced_features = tsne.fit_transform(features)

        # Remove hook
        hook_handle.remove()

        return reduced_features


def hooker(t):
    """
    Make a hook function which appends model output to the given list.

    Keep in mind that assigning an existing list to a variable
    actually provides a "pointer" to the list
    """

    def function(module, args, output):
        t.append(output.detach().cpu().numpy())
        print("From hook, latest append:", t[-1].shape)

    return function


def t_sne(features, dimensionality):
    perplexity_n = min(30, len(features) - 1)
    np.random.seed(
        42
    )  # @Wilhelmsen: Define seed elsewhere, once data has been visualized to graph
    tsne = TSNE(n_components=dimensionality, perplexity=perplexity_n)
    data = tsne.fit_transform(features)

    return data


def reduce_data(trained_file, categories, target_dimensionality=2, method="_tSNE"):
    """Take a homogenous array of data, and reduce its dimensionality through t-SNE."""
    """Deprecated"""
    # TEMP: This is a hard-coded simulation of choosing a discrete layer
    model_obj = AutoencodeModel(categories, trained_file)
    model_obj.load_from_checkpoint()
    model_obj.to(model_obj.device)
    model_obj.eval()

    features = []

    # @Wilhelmsen: What *is* the output?
    def my_hook(model, args, output):
        features.append(output.detach())

    # def hooker(d, keyname):
    #    def hook(model, args, output):
    #        d[keyname] = output.detach()
    #    return hook

    # Gets the "learnable parameters" from the model's state_dict
    parameters = list(model_obj.state_dict().values())
    # Selects a small slice of the parameters to t-SNE
    selected_features = parameters[163:167]
    selected_features = np.array(selected_features)

    # Else try layer4.0
    # Notice that the layer is here: ~~~~~~vvvvvv
    hook_handle = model_obj.model.backbone.layer4.register_forward_hook(my_hook)

    # Forward the model
    # with torch.no_grad():
    #    _ = model_obj()

    # print("Number of entries in features:", len(features))

    hook_handle.remove()

    # Reduce dimensionality by t-SNE
    """ perplexity_n = min(30, len(selected_features) - 1)
    np.random.seed(42)  # @Wilhelmsen: Define seed elsewhere, once data has been visualized to graph
    tsne = TSNE(n_components=target_dimensionality, perplexity=perplexity_n)
    reduced_data = tsne.fit_transform(selected_features) """

    # Added a switch for later implementation of more reduction methods
    match method:
        case "_tSNE":  # Maybe have this be based off of enums  instead?
            reduced_data = t_sne(selected_features, target_dimensionality)
        case _:  # Default case
            reduced_data = False
            print("Error: No reduction method selected!")

    if reduced_data:
        return reduced_data


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
