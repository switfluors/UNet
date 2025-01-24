import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metadata
DATE = '01232025'
ITERATION = 0

# Data settings
NORMALIZATION = False
DENORMALIZATION = False
TRAINING_DATA_PATH = "../data/Training_Perlin40k_Pperlin50k_mat2gray.mat"
TESTING_DATA_PATH = "../data/Testing_Perlin5k_Pperlin50k_mat2gray_2.mat"
TRAINING_DATASET_SIZE = 40000
TESTING_DATASET_SIZE = 5000
TRAIN_TEST_SPLIT = 0.80
BATCH_SIZE = 50
NUM_WORKERS = 4  # Adjust based on your system's resources
INPUT_SIZE = (1, 16, 128)

# Model settings
MODEL_TYPE = "attention_unet"  # Options: "unet", "attention_unet"
NUM_LAYERS = 4
NUM_FIRST_FILTERS = 4

# Training settings
EPOCHS = 250
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
LOSS_FUNCTION = "mse"  # Options: "mse", "mae"
OPTIMIZER = "adam"  # Options: "adam", "sgd"
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.2

# Seed for reproducibility
SEED = 42