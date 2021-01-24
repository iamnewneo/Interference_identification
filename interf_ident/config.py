import os

CLASSES = ["CI", "CWI", "MCWI", "SOI"]
N_FOLDS = 5
SEED = 42
BASE_PATH = os.getenv("BASE_PATH", ".")
# DEVICE = "cuda"
# BATCH_SIZE = 100
# MAX_EPOCHS = 25
DEVICE = "cpu"
BATCH_SIZE = 5
MAX_EPOCHS = 3
LR = 1e-3
