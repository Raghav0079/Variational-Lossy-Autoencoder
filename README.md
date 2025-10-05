# Variational Lossy Autoencoder (VLAE) with Conditional Prior

This repository contains an implementation of the Variational Lossy Autoencoder (VLAE) for the MNIST dataset, featuring a conditional prior. The project explores lossy compression and generative modeling using deep variational inference, following the approach described in the paper "Variational Lossy Autoencoder" (Chen et al., 2016).

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Files and Directories](#files-and-directories)
- [References](#references)
- [License](#license)

## Overview

Variational Autoencoders (VAEs) are generative models that learn a probabilistic mapping from data to a latent space. The VLAE extends the VAE by introducing a lossy compression mechanism and a more expressive prior, allowing for better modeling of complex data distributions. In this project, we implement a VLAE with a conditional prior for the MNIST handwritten digit dataset.

Key features:
- Conditional prior based on class labels
- Lossy compression via variational inference
- Training and evaluation on MNIST
- Visualization of generated samples and reconstructions

## Project Structure
