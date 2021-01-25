import os

CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
N_FOLDS = 5
SEED = 42
BASE_PATH = os.getenv("BASE_PATH", ".")
LR = 1e-3
DEVICE = "cuda"
BATCH_SIZE = 100
MAX_EPOCHS = 25
# DEVICE = "cpu"
# BATCH_SIZE = 5
# MAX_EPOCHS = 3