# Latent Space Visualizer

Developed by Olivia Linnea Kopsland Tjore & William Westye Mikal Wilhelmsen


## Build Instructions (Style 1)

A simple command-by-command guide to installing and running the program from a unix system with [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Python](https://www.python.org/downloads/) already installed.

```sh
$ # Clone repository and navigate inside
$ git clone https://github.com/squonk4303/latent-space-visualizer
$ cd latent-space visualizer

$ # Make and source a python virtual environment
$ python -m venv .venv
$ source .venv/bin/activate

$ # Install dependencies
$ pip install -r requirements.txt
$ # torch and torchvision have to be installed separately;
$ # see instructions at https://pytorch.org/get-started/locally/

$ # Run the program as a python package
$ python -m visualizer.main
```

## Build Instructions (Style 2)

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

Note that `torch` and `torchvision` have to be installed separately; see [their official instructions](https://pytorch.org/get-started/locally/)
```sh
pip install -r requirements.txt
```

Run the program as a python package, commands are supported
```sh
python -m visualizer.main
python -m visualizer.main --help
```
