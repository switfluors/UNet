import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
import config
import os

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

def get_train_test_datasets(args):

    NOISE_SCALE_STR = f"_Scale{args.noise_scale}_" if args.noise_scale != 1 else "_"

    TRAINING_DATA_PATH = (f"../data/Training_{args.noise_type}{args.train_dataset_size // 1000}k_"
                          f"P{args.noise_type.lower()}{args.noise_level // 1000}k{NOISE_SCALE_STR}"
                          f"{config.NORMALIZATION_TECH}.mat")
    TESTING_DATA_PATH = (f"../data/Testing_{args.noise_type}{args.test_dataset_size // 1000}k_"
                         f"P{args.noise_type.lower()}{args.noise_level // 1000}k{NOISE_SCALE_STR}"
                         f"{config.NORMALIZATION_TECH}.mat")

    with h5py.File(TRAINING_DATA_PATH) as matdata_train:

        sptimg4_train_notnorm = matdata_train['sptimgPerlin'][:]
        tbg4_train_notnorm = matdata_train['tbgPerlin'][:]
        GTspt_train_notnorm = matdata_train['GTspt'][:]

        if config.NORMALIZATION:
            sptimg4_train_norm = matdata_train['normalized_sptimgPerlin'][:]
            tbg4_train_norm = matdata_train['normalized_tbgPerlin'][:]
            GTspt_train_norm = matdata_train['normalized_GTspt'][:]

    sptimg4_train_notnorm = np.expand_dims(sptimg4_train_notnorm, axis=1)
    tbg4_train_notnorm = np.expand_dims(tbg4_train_notnorm, axis=1)
    GTspt_train_notnorm = np.expand_dims(GTspt_train_notnorm, axis=1)

    if config.NORMALIZATION:
        sptimg4_train_norm = np.expand_dims(sptimg4_train_norm, axis=1)
        tbg4_train_norm = np.expand_dims(tbg4_train_norm, axis=1)
        GTspt_train_norm = np.expand_dims(GTspt_train_norm, axis=1)


    if args.test_dataset_size == 0:
        if config.OUTPUT_SPECTRA:
            if config.NORMALIZATION:
                data = CustomDataset(sptimg4_train_norm, GTspt_train_norm)
            else:
                data = CustomDataset(sptimg4_train_notnorm, GTspt_train_notnorm)
        else:
            if config.NORMALIZATION:
                data = CustomDataset(sptimg4_train_norm, tbg4_train_norm)
            else:
                data = CustomDataset(sptimg4_train_notnorm, tbg4_train_notnorm)

        train_size = round(args.train_test_split * len(data))
        test_size = len(data) - train_size

        train_dataset, test_dataset = random_split(data, [train_size, test_size])

    else:
        with h5py.File(TESTING_DATA_PATH, 'r') as matdata:
            sptimg4_test_notnorm = matdata['sptimgPerlin'][:]

            if config.NORMALIZATION:
                sptimg4_test_norm = matdata['normalized_sptimgPerlin'][:]
                tbg4_test_norm = matdata['normalized_tbgPerlin'][:]
                tbg4_test_notnorm = matdata['tbgPerlin'][:]
                gt_spt_test = matdata['normalized_GTspt'][:]
            else:
                tbg4_test_notnorm = matdata['tbgPerlin'][:]
                gt_spt_test = matdata['GTspt'][:]

        sptimg4_test_notnorm = np.expand_dims(sptimg4_test_notnorm, axis=1)
        tbg4_test_notnorm = np.expand_dims(tbg4_test_notnorm, axis=1)
        gt_spt_test = np.expand_dims(gt_spt_test, axis=1)

        if config.NORMALIZATION:
            sptimg4_test_norm = np.expand_dims(sptimg4_test_norm, axis=1)
            tbg4_test_norm = np.expand_dims(tbg4_test_norm, axis=1)

        if config.OUTPUT_SPECTRA:
            if config.NORMALIZATION:
                train_dataset = CustomDataset(sptimg4_train_norm, gt_spt_test)
                test_dataset = CustomDataset(sptimg4_test_norm, gt_spt_test)
            else:
                train_dataset = CustomDataset(sptimg4_train_notnorm, gt_spt_test)
                test_dataset = CustomDataset(sptimg4_test_notnorm, gt_spt_test)

        else:
            if config.NORMALIZATION:
                train_dataset = CustomDataset(sptimg4_train_norm, tbg4_train_norm)
                test_dataset = CustomDataset(sptimg4_test_norm, tbg4_test_norm)
            else:
                train_dataset = CustomDataset(sptimg4_train_notnorm, tbg4_train_notnorm)
                test_dataset = CustomDataset(sptimg4_test_notnorm, tbg4_test_notnorm)


    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=True)

    return train_loader, test_loader, gt_spt_test




