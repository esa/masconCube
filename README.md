# masconCube

## Installation

```bash
conda env create -f environment.yml
conda activate masconcube
```

Or, if you want to install also development dependencies:

```bash
conda env create -f environment_dev.yml
conda activate masconcube
```

Development dependencies include packages for linting and contributing to the project, as well as libraries to work with
meshes and 3D data, that are used for data generation and visualization, but are not strictly necessary to run training
and inference.