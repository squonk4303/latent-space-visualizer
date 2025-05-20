# Latent Space Visualizer

A GUI tool for visualizing the latent space of pre-trained variational autoencoders (VAEs) for semantic segmentation applications. Providing simple to use methods for loading and processing user selected models and datasets. Developed as part of a Bachelor's project. Designed to be open and evolved by the community.

Developed by Olivia Linnea Kopsland Tjore & William Westye Mikal Wilhelmsen

## Build Instructions

A simple command-by-command guide to installing and running the program from a unix system with [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Python](https://www.python.org/downloads/) already installed.

Clone repository and navigate inside
```sh
git clone https://github.com/squonk4303/latent-space-visualizer
cd latent-space visualizer
```

Make and source a python virtual environment
```sh
python -m venv .venv
source .venv/bin/activate
```

Install dependencies.

Note that [torch and torchvision](https://pytorch.org/get-started/locally/) have to be installed separately.
```sh
pip install -r requirements.txt
```

Run the program as a python package, commands are supported
```sh
python -m visualizer.main
python -m visualizer.main --help
```
