[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Uncertainty Quantification Reveals the Limits of Lattice Thermal Conductivity Prediction

This repository contains the code and data to reproduce the results presented in the paper **"Uncertainty Quantification Reveals the Limits of Lattice Thermal Conductivity Prediction"**.

### Datasets
All datasets used in this study are available through their respective source links:

* **Starrydata**: Accessible via [Figshare](https://figshare.com/projects/Starrydata_datasets/155129).
* **Citrine & UCSB**: Obtained using the [`matminer`](https://github.com/hackingmaterials/matminer) package.
* **ESTM**: Sourced from the original [KRICT-DATA/SIMD GitHub Repository](https://github.com/KRICT-DATA/SIMD).
* **CHER**: Retrieved directly from the supporting information of the original publication.
* **Itani et al.**: Downloaded from [NEMAD](https://nemad.org/).

## Citation
For now, please cite the ChemRxiv version.

## Contact
Milad Rayka, milad.rayka@yahoo.com

## Installation

### Prerequisites
If you do not have Mamba installed, the fastest way is to install **Miniforge** (which includes Mamba by default):

* **Windows / macOS / Linux**: Download and run the installer from the [Miniforge GitHub Repository](https://github.com/conda-forge/miniforge).

Alternatively, if you already have Conda installed, you can add Mamba to your base environment:
```bash
conda install -c conda-forge mamba

```

---

### Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/miladrayka/uncertainty_quantification.git
cd uncertainty_quantification

```

2. **Create and activate the Mamba environment:**
Using the provided `environment.yml` file:
```bash
mamba env create -f environment.yml
mamba activate ml

```

## Usage

To reproduce all results, tables, and figures, refer to the *./evidential_regression_package/workflow.ipynb* and *./mc_dropout_package/workflow.ipynb* folders.

## Copyright

Copyright (c) 2026, Milad Rayka.
