# This file contains the configuration for the models and data loaders

import torch

# Data paths
TRAIN_IMAGE_DIR = "./data/train/train"
TRAIN_MASK_DIR = "./data/train_gt/train_gt"
TEST_IMAGE_DIR = "./data/test/test"

# Model parameters
IN_CHANNELS = 3
OUT_CHANNELS = 3
NUM_LAYERS = 2
BASE_CHANNELS = 64
KERNEL_SIZE = 3
PADDING = 1
DROPOUT_RATE = 0.2
UPSAMPLING_METHOD = "TransposedConv"

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data parameters
IMG_SIZE = (256, 256)
AUGMENTATIONS = False
VAL_SPLIT = 0.15

# Additional parameters
EXPERIMENT_NAME = "experiment_0"