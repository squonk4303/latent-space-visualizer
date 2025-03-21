#!/usr/bin/env python3
from sklearn.manifold import TSNE
import mmap
import numpy as np
import PIL
import tempfile
import torch
import torchvision

from visualizer import consts
from visualizer.external import fcn


class AutoencodeModel:  # @Wilhelmsen: methinks this can be renamed to "ModelManager"
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.models = {}  # @Wilhelmsen: Perhaps there should be one model per object
        # @Wilhelmsen make the transformed image size a const, maybe choosable
        self.preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(640),
                torchvision.transforms.ToTensor(),
            ]
        )

    def load_model(self, trained_file, categories) -> None:
        """Load trained model from file to local dictionary."""
        # Create model and load data to memory
        # @Wilhelmsen: Alter to include more models, when we include more models
        # if it's possible to do it in the same function...
        model_obj = fcn.FCNResNet101(categories)
        checkpoint = torch.load(
            trained_file, map_location=self.device, weights_only=False
        )

        # Make necessary alterations to state_dict before loading into model
        # @Wilhelmsen: This can surely be foreshortened, perhaps with list comprehension...?
        state_dict = checkpoint["state_dict"]
        new_state_dict = dict()
        for key, value in state_dict.items():
            new_key = key.removeprefix("module.")
            new_state_dict[new_key] = value

        checkpoint["state_dict"] = new_state_dict
        model_obj.load_state_dict(checkpoint["state_dict"], strict=True)

        model_obj.to(self.device)
        model_obj.eval()

        return model_obj

    # TODO: @Test
    def dataset_to_tensors(self, image_paths: list):
        """
        TODO: Uses the file dialog to locate a dir (or zip file maybe) in
        which to scan for valid images and load them into memory. Also make
        an effort to state how many images are loaded, because *that* has a
        lot of implications that are relevant to the user.

        @Wilhelmsen: Could this and single_image_to_tensor be the same function?
        """
        # Open images for processing with PIL.Image.open
        # @Wilhelmsen: PIL.Image.open opens the file and it remains open until the data is processed
        # Perhaps there's some optimization to be done?
        dataset = [PIL.Image.open(image).convert("RGB") for image in image_paths]

        # @Wilhelmsen: ?
        # Save it as an attribute, because we want to append-to and reuse these
        tensors = [
            self.preprocessing(img).unsqueeze(0).to(self.device) for img in dataset
        ]

        return tensors

    def preliminary_dim_reduction(self, model, image_tensors, layer):
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
                        feature_map, dtype=torch.float32, device=self.device
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

    def apply_tsne(self, features):
        """Makes a t-SNE object based on input and applies it to supplied features."""
        # Ensure a reasonable/legal perplexity value
        perplexity_value = min(30, len(features) - 1)

        # Define and apply t-SNE
        tsne = TSNE(
            n_components=2, perplexity=perplexity_value, random_state=consts.SEED
        )
        reduced_features = tsne.fit_transform(features)

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
