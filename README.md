[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license: GPLv3](https://img.shields.io/badge/license-GPLv3-red)](https://www.gnu.org/licenses/gpl-3.0.en.html)
[![made with: PyQt6](https://img.shields.io/badge/made_with-PyQt6-2CDE85)](https://www.riverbankcomputing.com/software/pyqt/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)


# Latent Space Visualizer

A GUI tool for visualizing the latent space of pre-trained variational autoencoders
(VAEs) for semantic segmentation applications. Provides simple-to-use interface to load
and process user selected models and datasets. Initially developed as part of a
Bachelor's project at the Norwegian University of Science and Technology (NTNU), in
collaboration with the Rochester Institute of Technology (RIT).

Designed to be open and evolved by the community.

Initially developed by Olivia Linnea Kopsland Tjore & William Westye Mikal Wilhelmsen


## Build Instructions

A simple command-by-command suggestion for installing and running the program from a
unix system with [Git][git-install] and [Python][python-install] already installed.

Clone repository and navigate inside the directory:

```sh
git clone https://github.com/squonk4303/latent-space-visualizer
cd latent-space-visualizer
```

Make and source a python virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
```

<br> Install dependencies:

(Please install torch and torchvision based on instructions from their [official
site][torch-install] for the best support for your GPU.)

```sh
pip install -r requirements.txt
pip install torch torchvision  # Installing from official site is recommended
```

Run the program as a python package. Commands are supported.

```sh
python -m visualizer.main
python -m visualizer.main --help
```


[git-install]:     https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
[python-install]:  https://www.python.org/downloads/
[torch-install]:   https://pytorch.org/get-started/locally/
