import torch
from datetime import datetime

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metadata
DATE = '02012025' # datetime.now().strftime("%m%d%Y")
ITERATION = 0

# Data settings
DATA_NOISE = "Perlin" # options: "Gaussian", "Perlin"
NORMALIZATION_TECH = "MatchNorm" # options: 'MatchNorm', 'mat2gray'
NORMALIZATION = True
OUTPUT_SPECTRA = True
TRAINING_DATA_PATH = "../data/Training_Perlin40k_Pperlin50k_MatchNorm.mat"
TESTING_DATA_PATH = "../data/Testing_Perlin5k_Pperlin50k_MatchNorm.mat"
TRAINING_DATASET_SIZE = 40000
TESTING_DATASET_SIZE = 5000
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