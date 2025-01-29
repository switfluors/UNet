import time
import numpy as np
import torch
import config
import argparse
import random
import os
from utils import ignore_warnings, set_seed, get_base_folder_name, get_model_path_filename
from models import UNet, AttentionUNet
from dataset import get_train_test_datasets
from train import train
from test import test

def get_models():
    """Initialize the chosen model."""
    models = {}
    if config.MODEL_TYPE not in ["unet", "attention_unet", "all"]:
        raise ValueError("Invalid model type: {config.MODEL_TYPE}")

    if config.MODEL_TYPE in ["unet", "all"]:
        models["unet"] = UNet(config.INPUT_SIZE[0], config.NUM_LAYERS, config.NUM_FIRST_FILTERS).to(config.DEVICE)
    if config.MODEL_TYPE in ["attention_unet", "all"]:
        models["attention_unet"] = AttentionUNet(config.INPUT_SIZE[0], config.NUM_LAYERS, config.NUM_FIRST_FILTERS).to(config.DEVICE)
    print(models.keys())
    return models

def main():

    ignore_warnings()

    parser = argparse.ArgumentParser(description="Train or Test a UNet Model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    args = parser.parse_args()

    # Ensure a valid mode is selected
    if not args.train and not args.test:
        parser.error("No action specified, add --train or --test")

    set_seed(config.SEED)

    os.chdir("test_models")

    if args.train:

        training_times = {}

        train_loader, test_loader, _ = get_train_test_datasets()
        models = get_models()

        base_folder = get_base_folder_name(config.MODEL_TYPE)
        os.makedirs(base_folder, exist_ok=True)
        os.chdir(base_folder)

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...\n")

            # Create subdirectory for each model
            os.makedirs(model_name, exist_ok=True)
            os.chdir(model_name)

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

            # Train model
            start_time = time.time()
            train(model_name, model, train_loader, test_loader, criterion, optimizer, scheduler)
            end_time = time.time()

            training_times[model_name] = end_time - start_time
            # Move back to base directory
            os.chdir("..")

        print(training_times)

        os.chdir("..")

    if args.test:
        _, test_loader, test_loader_notnorm, gt_spt_test = get_train_test_datasets()

        base_folder = get_base_folder_name(config.MODEL_TYPE)
        os.chdir(base_folder)

        models = get_models()
        for model_folder in models.keys():
            models[model_folder].load_state_dict(
                torch.load(os.path.join(model_folder, get_model_path_filename(model_folder)), weights_only=False))
        test(models, test_loader, test_loader_notnorm, gt_spt_test)

if __name__ == "__main__":
    main()