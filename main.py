import numpy as np
import torch
import config
import argparse
import random
import os
from utils import get_folder_name
from models import UNet, AttentionUNet
from dataset import get_train_test_datasets
from train import train
from test import test

def set_seed(seed=42):
    """Ensures reproducibility by setting random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model():
    """Initialize the chosen model."""
    if config.MODEL_TYPE == "unet":
        return UNet(config.INPUT_SIZE[0], config.NUM_LAYERS, config.NUM_FIRST_FILTERS).to(config.DEVICE)
    elif config.MODEL_TYPE == "attention_unet":
        return AttentionUNet(config.INPUT_SIZE[0], config.NUM_LAYERS, config.NUM_FIRST_FILTERS).to(config.DEVICE)
    else:
        raise ValueError(f"Invalid model type: {config.MODEL_TYPE}")

def main():
    parser = argparse.ArgumentParser(description="Train or Test a UNet Model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    args = parser.parse_args()

    # Ensure a valid mode is selected
    if not args.train and not args.test:
        parser.error("No action specified, add --train or --test")

    set_seed(config.SEED)

    foldername = get_folder_name()

    if args.train:
        os.makedirs(foldername, exist_ok=True)
        os.chdir(foldername)

        train_loader, test_loader = get_train_test_datasets()

        # Load model
        model = get_model()

        # Define loss function
        if config.LOSS_FUNCTION == "mse":
            criterion = torch.nn.MSELoss()
        elif config.LOSS_FUNCTION == "mae":
            criterion = torch.nn.L1Loss()
        else:
            raise ValueError(f"Invalid loss function: {config.LOSS_FUNCTION}")

        # Define optimizer
        if config.OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        elif config.OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Invalid optimizer: {config.OPTIMIZER}")

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)

        train(model, train_loader, test_loader, criterion, optimizer, scheduler)
        # test(model, test_loader, criterion, config)

if __name__ == "__main__":
    main()