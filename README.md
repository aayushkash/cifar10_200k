# CIFAR10 Classification with Custom CNN

This repository contains a PyTorch implementation of a custom CNN architecture for CIFAR10 classification with the following specifications:

## Model Architecture
- Uses C1C2C3C4 architecture without MaxPooling
- Implements Depthwise Separable Convolution
- Uses Dilated Convolution
- Global Average Pooling (GAP)
- Total parameters < 200k
- Receptive Field > 44

## Data Augmentation
Using Albumentations library with:
- Horizontal Flip
- ShiftScaleRotate
- CoarseDropout

## Project Structure 
- `models/model.py`: Contains the custom CNN architecture.
- `train.py`: Training script.
- `test.py`: Testing script.
- `data/`: Contains the CIFAR10 dataset.
- `utils/`: Contains utility functions.
- `README.md`: This file.

## Achieved desired accuracy
- 85.x%
