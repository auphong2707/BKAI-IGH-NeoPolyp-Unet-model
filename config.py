# This file contains the configuration for the models and data loaders

import torch

# Model parameters
IN_CHANNELS = 3
OUT_CHANNELS = 3
NUM_LAYERS = 6
BASE_CHANNELS = 128
KERNEL_SIZE = 3
PADDING = 1
DROPOUT_RATE = 0.3
UPSAMPLING_METHOD = "TransposedConv"

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data parameters
IMG_SIZE = (512, 512)
AUGMENTATIONS = True
VAL_SPLIT = 0.15

# Additional parameters
EXPERIMENT_NAME = "experiment_0"