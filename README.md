# Analyzing and Comparing Performance of Spatial Attention to UNet Structure for Single-Molecule Microscopy Spectral Image Denoising

This project implements a deep learning model implementing a spatial attention layer to each of the upsampling layers
within the traditional UNet architecture for processing spectral images. The goal is to predict the background
(and thereby) the spectra information using simulated Perlin and Gaussian noise spectral images. We demonstrate that
this implementation can produce significant improvements both quantitatively (~30-40% RMSE reduction) and qualitatively.
Moreover, our project implementation is highly configurable, as it supports training/testing with the same or different
datasets, changing hyperparameters, and more.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Contributions](#contributions)
5. [Acknowledgments](#acknowledgments)

## Project Structure

├── data/                    # Folder to store all datasets
├── test_models/             # Folder for saving trained models and test results
├── README.md                # Project documentation (This file)
├── main.py                  # Main script to run training or testing
├── config.py                # Configuration settings, hyperparameters, and paths
├── dataset.py               # Data loading and preprocessing functions
├── train.py                 # Training script
├── test_background.py       # Testing script for background predictions
├── test_spectra.py          # Testing script for spectral predictions
├── utils.py                 # Utility functions (e.g., setting random seed, logging)
├── models.py                # Model definitions (Conventional UNet, Spatial Attention UNet)
├── hyperparameter_search.py # Test various hyperparameter combinations to determine best-performing variation
├── requirements.txt         # List of dependencies
└── .gitignore               # Git ignore file to exclude unnecessary files

## Setup

## Usage

## Contributions

## Acknowledgments
