
# Variational Lossy Autoencoder (VLAE) with Conditional Prior

This repository provides an implementation of the Variational Lossy Autoencoder (VLAE) for the MNIST dataset, featuring a conditional prior. The project explores lossy compression and generative modeling using deep variational inference, following the approach described in the paper "Variational Lossy Autoencoder" (Chen et al., 2016).

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

Variational Autoencoders (VAEs) are generative models that learn a probabilistic mapping from data to a latent space. The VLAE extends the VAE by introducing a lossy compression mechanism and a more expressive, possibly conditional, prior, allowing for better modeling of complex data distributions. This implementation uses a conditional prior based on MNIST class labels.

**Key features:**
- Conditional prior based on class labels
- Lossy compression via variational inference
- Training and evaluation on MNIST
- Visualization of generated samples and reconstructions

## Project Structure

```text
.
├── code/
│   ├── vlae_mnist_conditional_prior.py   # Main VLAE implementation
│   └── VLAE.tex                          # LaTeX source for report/paper
│
├── data/
│   ├── train-images-idx3-ubyte(.gz)      # MNIST training images
│   ├── train-labels-idx1-ubyte(.gz)      # MNIST training labels
│   ├── t10k-images-idx3-ubyte(.gz)       # MNIST test images
│   └── t10k-labels-idx1-ubyte(.gz)       # MNIST test labels
│
├── result/
│   ├── sample_conditioned_epoch*.png     # Generated samples per epoch
│   ├── test_recon_epoch*.png             # Reconstructions per epoch
│   └── vlae_condprior_epoch*.pt          # Model checkpoints
│
└── VLAE.pdf                              # Project report (compiled)
```

## Installation

1. **Clone the repository:**
	 ```powershell
	 git clone https://github.com/yourusername/Variational-Lossy-Autoencoder.git
	 cd Variational-Lossy-Autoencoder
	 ```
2. **Install dependencies:**
	 ```powershell
	 pip install torch numpy matplotlib
	 ```
	 - Python 3.7+
	 - PyTorch (tested with 1.7+)
	 - NumPy
	 - Matplotlib

3. **Download MNIST data:**
	 - Download the four MNIST files from [the official website](http://yann.lecun.com/exdb/mnist/) and place them in the `data/` directory. Both compressed (`.gz`) and uncompressed files are supported.

## Usage

To train the VLAE on MNIST with a conditional prior:

```powershell
python code/vlae_mnist_conditional_prior.py
```

- Training progress, loss values, and sample images will be saved in the `result/` directory.
- Model checkpoints are saved as `vlae_condprior_epoch{N}.pt`.
- Generated samples and reconstructions are saved as PNG images for each epoch.

### Customization

You can modify the script to accept command-line arguments for hyperparameters such as batch size, learning rate, number of epochs, etc. (see code for details or extend as needed).

## Training Details

- **Dataset:** MNIST (60,000 train, 10,000 test images)
- **Model:** VLAE with conditional prior (class label as condition)
- **Loss:** Variational lower bound (reconstruction + KL divergence)
- **Optimizer:** Adam
- **Epochs:** 30 (default)
- **Batch size:** 128 (default)

## Results

- **Sample Images:**
	- `result/sample_conditioned_epoch{N}.png` — Samples generated from the model at epoch N, conditioned on class labels.
- **Reconstructions:**
	- `result/test_recon_epoch{N}.png` — Reconstructions of test images at epoch N.
- **Model Checkpoints:**
	- `result/vlae_condprior_epoch{N}.pt` — Saved PyTorch model state dicts for each epoch.

## References

- Chen, X., Kingma, D. P., Salimans, T., Duan, Y., Dhariwal, P., Schulman, J., ... & Abbeel, P. (2016). [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731). arXiv preprint arXiv:1611.02731.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## License

This project is provided for research and educational purposes only. Please cite the original paper if you use this code or results in your work.
