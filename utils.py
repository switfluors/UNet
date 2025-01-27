import warnings
import torch
import numpy as np
import random
import config

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

def get_base_folder_name(model_name):
    model_name_folder = None
    if model_name == 'unet':
        model_name_folder = "conv_UNet_model"
    elif model_name == 'attention_unet':
        model_name_folder = "conv_att_UNet_model"
    elif model_name == 'all':
        model_name_folder = "all_models"

    foldername = (f'OG_Perlin_mat2gray_{"norm" if config.NORMALIZATION else "notnorm"}' +
                       f'{"_denorm" if config.DENORMALIZATION else "_"}{str(int(config.TRAINING_DATASET_SIZE / 1000))}' +
                       f'k_{model_name_folder}_{str(config.EPOCHS)}epo_' +
                       f'{str(config.LEARNING_RATE)}lr_{str(config.WEIGHT_DECAY)}l2reg_bs{str(config.BATCH_SIZE)}' +
                       f'_stepsize{str(config.SCHEDULER_STEP_SIZE)}_' +
                       f'lrdecay{str(config.SCHEDULER_GAMMA)}_{str(config.TRAIN_TEST_SPLIT)}pc_train_split_{config.DATE}_'
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