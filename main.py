import time
import torch
import config
import argparse
import os
from utils import ignore_warnings, set_seed, get_base_folder_name, get_model_path_filename, get_models, log_print
from dataset import get_train_test_datasets
from train import train
import test_background
import test_spectra
from logger import get_logger

def main():

    ignore_warnings()

    parser = argparse.ArgumentParser(description="Train or Test a UNet Model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--epochs", default=250, type=int, help="Number of epochs")
    parser.add_argument("--bs", default=50, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.01 if not config.NORMALIZATION else 0.001, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.01 if not config.NORMALIZATION else 0.0001, type=float, help="Learning rate decay")
    parser.add_argument("--loss_fn", default="mse", type=str, help="Loss function")
    parser.add_argument("--optimizer", default="adam", type=str, help="Optimizer")
    parser.add_argument("--scheduler_step_size", default=50, type=int, help="Scheduler step size")
    parser.add_argument("--scheduler_gamma", default=0.2, type=float, help="Schedule gamma")
    parser.add_argument("--train_dataset_size", required=True, type=int, help="Train dataset size")
    parser.add_argument("--test_dataset_size", default=5000, type=int, help="Test dataset size")
    parser.add_argument("--train_test_split", default=0, type=float, help="Train test split")
    parser.add_argument("--noise_level", required=True, type=int, help="Dataset size noise level")
    parser.add_argument("--noise_type", default="Perlin", type=str, help="Noise type")
    parser.add_argument("--noise_scale", default=1, type=int, help="Noise scale factor")
    parser.add_argument("--model_type", default="attention_unet", type=str, help="Model type (attention_unet, unet, all)")
    args = parser.parse_args()

    # Ensure a valid mode is selected
    if not args.train and not args.test:
        parser.error("No action specified, add --train or --test")

    if args.train_test_split < 0 or args.train_test_split > 1:
        parser.error("Train and test split must be between 0 and 1")

    if (args.train_test_split > 0 and args.test_dataset_size > 0) or (args.train_test_split == 0 and args.test_dataset_size == 0):
        parser.error("You must choose either a test dataset size of 0 or train_test_split of 0")

    set_seed(config.SEED)

    os.chdir("test_models")

    if args.train:

        training_times = {}

        train_loader, test_loader, _ = get_train_test_datasets(args)
        models = get_models(args)

        base_folder = get_base_folder_name(args.model_type, args)
        os.makedirs(base_folder, exist_ok=True)
        os.chdir(base_folder)
        logger = get_logger("Main")
        log_print(logger, "Creating folder for training: ", base_folder)

        for model_name, model in models.items():
            log_print(logger, f"\nTraining {model_name}...\n")

            # Create subdirectory for each model
            os.makedirs(model_name, exist_ok=True)
            os.chdir(model_name)

            # Define loss function
            if args.loss_fn == "mse":
                criterion = torch.nn.MSELoss()
            elif args.loss_fn == "mae":
                criterion = torch.nn.L1Loss()
            else:
                raise ValueError(f"Invalid loss function: {args.loss_fn}")

            # Define optimizer
            if args.optimizer == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                raise ValueError(f"Invalid optimizer: {args.optimizer}")

            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

            # Train model
            start_time = time.time()
            train(model_name, model, train_loader, test_loader, criterion, optimizer, scheduler, logger, args)
            end_time = time.time()

            training_times[model_name] = end_time - start_time
            # Move back to base directory
            os.chdir("..")

        log_print(logger, f"Training times: {training_times}")

        os.chdir("..")

    if args.test:

        _, test_loader, gt_spt_test = get_train_test_datasets(args)

        base_folder = get_base_folder_name(args.model_type, args)
        os.chdir(base_folder)

        print("Accessing test folder: {}".format(base_folder))

        models = get_models(args)
        for model_folder in models.keys():
            models[model_folder].load_state_dict(
                torch.load(os.path.join(model_folder, get_model_path_filename(model_folder)), weights_only=False))
        if config.OUTPUT_SPECTRA:
            test_spectra.test(models, test_loader, args)
        else:
            test_background.test(models, test_loader, gt_spt_test, args)



if __name__ == "__main__":
    main()
