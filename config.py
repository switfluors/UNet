import torch
from datetime import datetime

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metadata
DATE = datetime.now().strftime("%m%d%Y")
ITERATION = 1

# Data settings
DATA_NOISE = "Perlin" # options: "Gaussian", "Perlin"
NORMALIZATION_TECH = "MatchNorm" # options: 'MatchNorm', 'mat2gray'
NORMALIZATION = True
OUTPUT_SPECTRA = False
TRAINING_DATASET_SIZE = 80000
P_NOISE = 100000
TESTING_DATASET_SIZE = 5000
TRAINING_DATA_PATH = f"../data/Training_{DATA_NOISE}{TRAINING_DATASET_SIZE // 1000}k_Pperlin{P_NOISE // 1000}k_{NORMALIZATION_TECH}.mat"
TESTING_DATA_PATH = f"../data/Testing_{DATA_NOISE}{TESTING_DATASET_SIZE // 1000}k_Pperlin{P_NOISE // 1000}k_{NORMALIZATION_TECH}.mat"
TRAIN_TEST_SPLIT = 0.80
BATCH_SIZE = 50
NUM_WORKERS = 4  # Adjust based on your system's resources
INPUT_SIZE = (1, 16, 128)

# Model settings
MODEL_TYPE = "all"  # Options: "unet", "attention_unet", "all"
NUM_LAYERS = 4
NUM_FIRST_FILTERS = 4

# Training settings
EPOCHS = 250
LEARNING_RATE = 0.01 if not NORMALIZATION else 0.001
WEIGHT_DECAY = 0.01 if not NORMALIZATION else 0.0001
LOSS_FUNCTION = "mse" if not NORMALIZATION else "mae" # Options: "mse", "mae"
OPTIMIZER = "adam"  # Options: "adam", "sgd"
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.2

# Seed for reproducibility
SEED = 42