import os

ENV = os.getenv("ENV", "dev")
configs = {
    "dev": {"DEVICE": "cpu", "BATCH_SIZE": 4, "MAX_EPOCHS": 3, "LR": 1,},
    "prod": {"DEVICE": "cuda", "BATCH_SIZE": 256, "MAX_EPOCHS": 200, "LR": 0.0001,},
}

SEED = 42
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

BASE_PATH = os.getenv("BASE_PATH", ".")

# Trainer
N_GPU = 1
N_WORKER = 8
FP_PRECISION = 16

# Training
DEVICE = configs[ENV]["DEVICE"]
BATCH_SIZE = configs[ENV]["BATCH_SIZE"]
MAX_EPOCHS = configs[ENV]["MAX_EPOCHS"]
LR = configs[ENV]["LR"]
