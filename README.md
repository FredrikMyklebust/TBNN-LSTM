# Tensor Basis Neural Networks for Unsteady Flow Prediction


## Project Overview

This repository contains code for training and evaluating Tensor Basis Neural Networks (TBNNs) for unsteady flow prediction. TBNNs incorporate physics-informed architecture to better represent and predict complex turbulent flow phenomena.


## Description

The project implements neural network models that leverage tensor basis formulations to ensure frame invariance and improved physical consistency when predicting turbulent flows. By incorporating tensor basis representations, the model respects fundamental physical principles that traditional neural networks might violate.


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
```

Install dependencies with:
```bash
pip install numpy scipy matplotlib torch pandas tqdm requests
```


## Data

The project uses turbulence statistics data from:
- **Source**: [Zenodo Dataset](https://zenodo.org/records/1095116)
- **Citation**: If you use this dataset, please cite the original authors

### Downloading Data
The repository includes code to automatically download and extract the required dataset:

```python
# Run the data download script in the notebook
# This will download the turbulence statistics dataset from Zenodo
```


## Project Structure
- `notebooks/`: Jupyter notebooks for data exploration and model training
  - `train_model.ipynb`: Main notebook for model training
- `data/`: Directory where downloaded and processed data is stored
- `models/`: Saved model weights and configurations


## Usage

### Training a Model
The main training pipeline is implemented in `notebooks/train_model.ipynb`:

1. The notebook handles data downloading, processing, and preparation
2. Uses an 80/20 train/validation split with weighted sampling biased toward near-wall regions
3. Implements a tensor basis neural network with physics-informed architecture
4. Trains the model with validation-based early stopping


## Model Architecture
The model incorporates tensor basis functions to represent the anisotropic Reynolds stress tensor. This approach:

1. Ensures Galilean invariance and coordinate frame invariance
2. Reduces the parameter space by leveraging known physical constraints
3. Improves generalization to unseen flow conditions


## License
[MIT License](LICENSE)


## Acknowledgments
This project uses turbulence statistics data from [Zenodo](https://zenodo.org/records/1095116), and we thank the original authors for making this data available.

