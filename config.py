import torch
import os

# base path of the dataset
DATASET_PATH = os.path.join("datasets", "train")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

VALIDATION_SPLIT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False


NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = 3

LR = 0.001
NUM_EPOCHS = 200
BATCH_SIZE = 1

INPUT_IMAGE_WIDTH = 2048
INPUT_IMAGE_HEIGHT = 2048

THRESHOLD = 0.5

BASE_OUTPUT = "output"

