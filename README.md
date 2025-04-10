# Tensor Basis Neural Networks for Unsteady Flow Prediction

## Project Overview

This repository contains code for training and evaluating Tensor Basis Neural Networks (TBNNs) for unsteady flow prediction. TBNNs incorporate physics-informed architectures to better represent and predict complex turbulent flow phenomena.

Additionally, this repository includes custom OpenFOAM turbulence models and utility solvers designed to compute and propagate model-form errors in Reynolds stress anisotropy and turbulent kinetic energy equations.

---

## Data

The project uses turbulence statistics data from:

**van der A, D.A., Scandura, P., O'Donoghue, T. (2018).**  
*Turbulence statistics in smooth wall oscillatory boundary layer flow*,  
**Journal of Fluid Mechanics**, *849*, 192–230.  
[DOI Link](https://doi.org/10.1017/jfm.2018.403)

- Dataset available on [Zenodo](https://zenodo.org/records/1095116)

If you use this dataset, please cite the original paper.

### Downloading Data

The repository includes code to automatically download and extract the dataset. You can run the download script from the provided notebook.

---

## Installation

### Requirements

```
numpy  
scipy  
matplotlib  
torch  
pandas  
tqdm  
requests  
fluidfoam
```

Install with:

```
pip install numpy scipy matplotlib torch pandas tqdm requests fluidfoam
```

---

## Project Structure

```
notebooks/               # Jupyter notebooks for training and analysis  
models/                  # Saved models and configurations  
data/                    # Folder for raw and processed data 
openfoam/                # Custom OpenFOAM turbulence models and scripts for OpenFOAM integration (e.g., calculateBdelta)
```

---

## OpenFOAM Models and Utilities

Custom OpenFOAM turbulence models are located in `openfoam/turbulenceModels/`. These models are designed to work with predictions from the TBNN and allow for calculating and propagating model-form errors in turbulent simulations.

### Models

- **frozennsolver**  
  Used to compute model-form errors in the Reynolds stress anisotropy (`b_ij`) and turbulent kinetic energy (`k`) equations.

- **frozenPropTime_stable**  
  Propagates predicted model-form errors from the TBNN through the flow field.

### Compilation

To compile the models:

1. Move the folders into your OpenFOAM `turbulenceModels/` directory.  
2. Compile using:

```
wmake libso
```

### `calculateBdelta`

A custom utility to compute the correction terms for you case.

> **Note**: Make sure to set the turbulence model to `frozennsolver` in your `constant/turbulenceProperties` file before using this utility.

---

## Adding Body Forces

To run a simulation with specified external body forces, follow these steps:

1. Clone and compile the body force module from [LiYZPearl/bodyforce](https://github.com/LiYZPearl/bodyforce):

```
git clone https://github.com/LiYZPearl/bodyforce.git  
cd bodyforce
```

2. Add the `filebasedbodyforce` source file to the bodyforce project before compiling. This module allows you to load time-varying body forces from an external file.

3. Compile the library:

```
wmake
```

4. Include the appropriate configuration in your OpenFOAM case files to activate the custom body force.

---

## Model Training

Model training is performed using the notebook:  
**`notebooks/train_model.ipynb`**

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Special thanks to the authors of the original turbulence dataset:

**van der A, D.A., Scandura, P., O'Donoghue, T. (2018)**  
*Turbulence statistics in smooth wall oscillatory boundary layer flow*,  
**Journal of Fluid Mechanics**, *849*, 192–230.
