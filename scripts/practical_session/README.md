# PyTorch MNIST Training Examples

Four Jupyter notebooks that show PyTorch basics and how to train models on MNIST. Shows underfitting, good fitting, and overfitting.

## Setup

Install conda environment:
```bash
conda env create -f env.yaml
conda activate pytorch-teaching
```

## Running

Start with the notebooks in order:
```bash
jupyter notebook
```

1. `00_datasets_and_dataloaders.ipynb` - Data loading basics
2. `01_pytorch_basics.ipynb` - PyTorch fundamentals  
3. `02_building_networks.ipynb` - Building neural networks
4. `03_mnist_training.ipynb` - Full training pipeline

## What's included

- **TooSimpleNet**: Single layer, underfits
- **GoodNet**: 2 hidden layers with dropout, fits well
- **TooComplexNet**: 11 layers, overfits badly
- **ConvNet**: CNN for comparison
