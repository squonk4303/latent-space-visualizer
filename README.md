[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GPLv3](https://img.shields.io/badge/License-GPLv3-red)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![PyQt6](https://img.shields.io/badge/Made%20With-PyQt6-blue)](https://www.riverbankcomputing.com/software/pyqt/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)



# Latent Space Visualizer

A GUI tool for visualizing the latent space of pre-trained variational autoencoders (VAEs) for semantic segmentation applications. Providing simple to use methods for loading and processing user selected models and datasets. Developed as part of a Bachelor's project at the Norwegian University of Science and Technology (NTNU), in collaboration with the Rochester Institute of Technology (RIT).

Designed to be open and evolved by the community.

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
