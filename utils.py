import warnings
import torch
import numpy as np
import random
import config
from models import UNet, AttentionUNet
import logging

def ignore_warnings():
    warnings.filterwarnings('ignore')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_base_folder_name(model_name, args):
    model_name_folder = None
    if model_name == 'unet':
        model_name_folder = "conv_UNet_model"
    elif model_name == 'attention_unet':
        model_name_folder = "conv_att_UNet_model"
    elif model_name == 'all':
        model_name_folder = "all_models"

    foldername = (f'OG_' +
                  f'{args.noise_type}_' +
                  f'{config.NORMALIZATION_TECH}_' +
                  f'{config.DATE}_' +
                  f'{"out_spectra" if config.OUTPUT_SPECTRA else "out_background"}_' +
                  f'{"norm" if config.NORMALIZATION else "notnorm"}_' +
                  f'{str(args.train_dataset_size // 1000)}k_' +
                  f'P{args.noise_type.lower()}{str(args.noise_level // 1000)}k_' +
                  f'{model_name_folder}_' +
                  f'{str(args.epochs)}epo_' +
                  f'{str(args.lr)}lr_' +
                  f'{str(args.weight_decay)}l2reg_' +
                  f'{str(args.bs)}bs_' +
                  f'{str(args.scheduler_step_size)}stepsize_' +
                  f'{str(args.scheduler_gamma)}lrdecay_' +
                  f'{args.loss_fn.upper()}loss_' +
                  f'{str(config.TRAIN_TEST_SPLIT)}pc_train_split_' +
                  f'{str(config.ITERATION)}')
    return foldername

def get_model_path_filename(model_name):
    if model_name == 'unet':
        model_name = "conv_UNet"
    elif model_name == 'attention_unet':
        model_name = "conv_att_UNet"

    model_filename = f'{model_name}_model.pth'

    return model_filename

    # model_filename = (f'OG_Perlin_mat2gray_{"norm" if config.NORMALIZATION else "notnorm"}_{model_name}_model' +
    #                       f'{"_denorm" if config.DENORMALIZATION else "_"}{str(int(config.TRAINING_DATASET_SIZE / 1000))}' +
    #                       f'k_{model_name}_model_{str(config.EPOCHS)}epo_' +
    #                       f'{str(config.LEARNING_RATE)}lr_{str(config.WEIGHT_DECAY)}l2reg_bs{str(config.BATCH_SIZE)}' +
    #                       f'_stepsize{str(config.SCHEDULER_STEP_SIZE)}_' +
    #                       f'lrdecay{str(config.SCHEDULER_GAMMA)}_{str(config.TRAIN_TEST_SPLIT)}pc_train_split_{config.DATE}_'
    #                       f'{str(config.ITERATION)}.pth')


def get_models():
    """Initialize the chosen model."""
    models = {}
    if config.MODEL_TYPE not in ["unet", "attention_unet", "all"]:
        raise ValueError("Invalid model type: {config.MODEL_TYPE}")

    if config.MODEL_TYPE in ["unet", "all"]:
        models["unet"] = UNet(config.INPUT_SIZE[0], config.NUM_LAYERS, config.NUM_FIRST_FILTERS).to(config.DEVICE)
    if config.MODEL_TYPE in ["attention_unet", "all"]:
        models["attention_unet"] = AttentionUNet(config.INPUT_SIZE[0], config.NUM_LAYERS, config.NUM_FIRST_FILTERS).to(config.DEVICE)
    models_string = ", ".join(models.keys())
    print("Available models: {}".format(models_string))
    return models

def log_print(logger, *args, **kwargs):
    """
    Prints and logs the message using the provided logger.
    """
    message = " ".join(str(arg) for arg in args)  # Convert args to string
    logger.info(message)  # Log the message
