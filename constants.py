## Configs for the project
# Model
MODEL_NAME = "bert-base-uncased"
ADAPTER_DIM = 64
ADAPTER_INIT_RANGE = 1e-2

# Data
DATASET_NAME = ("glue", "sst2")
BATCH_SIZE = 32
NUM_WORKERS = 2

# Training
SEED = 42
NUM_EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
ACCCELERATOR = "gpu"
DEVICES = 1
PRECISION = "16-mixed"

# Inference
CKPT_PATH = ""
