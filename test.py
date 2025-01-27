import torch
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import config


def test(models, test_loader, test_loader_notnorm):
    """Evaluates both models on the test dataset and stores results."""

    device = config.DEVICE

    print(models.keys())

    # Set models to evaluation mode
    for model in models.values():
        model.eval()

    sptimg4_test = test_loader.dataset.X
    tbg4_test = test_loader.dataset.Y

    print(sptimg4_test.shape)
    print(tbg4_test.shape)

    # Determine number of test samples
    numSpectra = len(test_loader.dataset)  # Get total dataset length

    # Preallocate arrays for predictions
    YPred = np.zeros((numSpectra, 1, 128, 16))
    Predictspe = np.zeros((numSpectra, 1, 128, 16))

    YPred2 = np.zeros((numSpectra, 1, 128, 16))
    Predictspe2 = np.zeros((numSpectra, 1, 128, 16))

    Oldsptimg = np.zeros((numSpectra, 1, 128, 16))

    # Compute mean background (assuming `tbg4_test` is from test dataset)
    mean_background = np.mean(tbg4_test, axis=0)
    print("Mean bg:", mean_background.shape)

    for n in range(numSpectra):
        input = torch.tensor(sptimg4_test[n:n + 1], dtype=torch.float32).to(device)

        conv_UNet_output = models["unet"](input).detach().cpu().numpy()
        YPred[n, :, :, :] = conv_UNet_output
        Predictspe[n, :, :, :] = sptimg4_test[n, :, :, :] - YPred[n, :, :, :]

        conv_att_UNet_output = models["attention_unet"](input).detach().cpu().numpy()
        YPred2[n, :, :, :] = conv_att_UNet_output
        Predictspe2[n, :, :, :] = sptimg4_test[n, :, :, :] - YPred2[n, :, :, :]

        Oldsptimg[n, :, :, :] = sptimg4_test[n, :, :, :] - mean_background

    YPred = np.squeeze(YPred, axis=1)
    Predictspe = np.squeeze(Predictspe, axis=1)
    YPred2 = np.squeeze(YPred2, axis=1)
    Predictspe2 = np.squeeze(Predictspe2, axis=1)
    Oldsptimg = np.squeeze(Oldsptimg, axis=1)
    mean_background = np.squeeze(mean_background, axis=0)


    sptn = np.mean(Predictspe[:, :, 6:10], axis=2)

    xq = np.arange(1, 128, 128 / 303)
    x = np.arange(1, 129)

    vq = np.zeros((numSpectra, 301))

    for i in range(sptn.shape[0]):
        # Create an interpolation function for the current row
        f = interp1d(x, sptn[i, :], kind='linear', fill_value="extrapolate")

        # Interpolate at the points in xq and store the result in vq
        vq[i, :] = f(xq)

    # print(vq.shape)

    sptimg4_test = np.squeeze(sptimg4_test, axis=1)
    rawspt = np.mean(sptimg4_test[:, :, 6:10], axis=2)
    rawspt_new = np.zeros((numSpectra, 301))

    for i in range(rawspt.shape[0]):
        f = interp1d(x, rawspt[i, :], kind='linear', fill_value="extrapolate")
        rawspt_new[i, :] = f(xq)

    rawspt = rawspt_new

    tbg4_test = np.squeeze(tbg4_test, axis=1)
    GTsptimg = sptimg4_test - tbg4_test
    GTspt = np.mean(GTsptimg[:, :, 6:10], axis=2)
    GTspt_new = np.zeros((numSpectra, 301))
    for i in range(GTspt.shape[0]):
        f = interp1d(x, GTspt[i, :], kind='linear', fill_value="extrapolate")
        GTspt_new[i, :] = f(xq)

    GTspt = GTspt_new

    Oldspt = np.mean(Oldsptimg[:, :, 6:10], axis=2)
    Oldspt_new = np.zeros((numSpectra, 301))
    for i in range(Oldspt.shape[0]):
        f = interp1d(x, Oldspt[i, :], kind='linear', fill_value="extrapolate")
        Oldspt_new[i, :] = f(xq)

    Oldspt = Oldspt_new

    def rmse(predictions, targets):
        """Calculate the RMSE between predictions and targets."""
        return np.sqrt(np.mean((predictions - targets) ** 2, axis=(1, 2)))

    ConvU_BG_RMSE = rmse(YPred, tbg4_test)
    print(ConvU_BG_RMSE.shape)
    ConvU_SPE_RMSE = rmse(Predictspe, GTsptimg)
    print(ConvU_SPE_RMSE.shape)
    ConvU_Att_BG_RMSE = rmse(YPred2, tbg4_test)
    print(ConvU_Att_BG_RMSE.shape)
    ConvU_Att_SPE_RMSE = rmse(Predictspe2, GTsptimg)
    print(ConvU_Att_SPE_RMSE.shape)

    print("Mean bg:", mean_background.shape)
    Old_BG = np.tile(mean_background[np.newaxis, :, :], (numSpectra, 1, 1))
    print(Old_BG.shape)
    Old_BG_RMSE = rmse(Old_BG, tbg4_test)
    print(Old_BG_RMSE.shape)
    Old_SPE_RMSE = rmse(Oldsptimg, GTsptimg)
    print(Old_SPE_RMSE.shape)

    # Plotting figures

    # First figure
    plt.figure(figsize=(12, 8))

    # Subplot titles
    titles = [
        'Conv. Att. U-Net Background RMSE',
        'Conv. Att. U-Net Spectra RMSE',
        'Conventional U-net Background RMSE',
        'Conventional U-net Spectra RMSE',
        'Old Method Background RMSE',
        'Old Method Spectra RMSE'
    ]

    # Create subplots for each RMSE
    for i, rmse_data in enumerate(
            [ConvU_Att_BG_RMSE, ConvU_Att_SPE_RMSE, ConvU_BG_RMSE, ConvU_SPE_RMSE, Old_BG_RMSE, Old_SPE_RMSE]):
        plt.subplot(3, 2, i + 1)
        plt.hist(rmse_data, bins=30, color='blue', alpha=0.7)
        plt.title(titles[i])
        plt.xlabel('RMSE Value')
        plt.ylabel('Counts')
        plt.grid(True)

    plt.suptitle('Comparison of RMSE Across Different Methods', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("rmse_figure_1.png")
    # plt.show()  # Display the figure

    # Second figure (with filtering)
    plt.figure(figsize=(12, 8))

    dataSets = [ConvU_Att_BG_RMSE, ConvU_Att_SPE_RMSE, ConvU_BG_RMSE, ConvU_SPE_RMSE, Old_BG_RMSE, Old_SPE_RMSE]
    filteredDataSets = [data[data <= 15] for data in dataSets]  # Filter values greater than 15

    # Define global bin edges for the range 0-15
    if not config.NORMALIZATION:
        binEdges = np.linspace(0, 15, 31)  # 30 bins from 0 to 15
    else:
        binEdges = np.linspace(0, 1, 31)

    # Loop through each filtered dataset to create subplot histograms with mean and std annotations
    for i, filtered_data in enumerate(filteredDataSets):
        plt.subplot(3, 2, i + 1)  # Set subplot position
        plt.hist(filtered_data, bins=binEdges, color='green', alpha=0.7)  # Plot histogram with defined bin edges

        meanVal = np.mean(filtered_data)
        stdVal = np.std(filtered_data)

        # Mean and standard deviation lines
        plt.axvline(meanVal, color='red', linestyle='-', linewidth=2, label=f'Mean: {meanVal:.2f}')
        plt.axvline(meanVal + stdVal, color='blue', linestyle='--', linewidth=2,
                    label=f'+1 Std: {meanVal + stdVal:.2f}')
        plt.axvline(meanVal - stdVal, color='blue', linestyle='--', linewidth=2,
                    label=f'-1 Std: {meanVal - stdVal:.2f}')

        plt.title(titles[i])
        plt.xlabel('RMSE Value')
        plt.ylabel('Counts')
        plt.legend()  # Show legend
        plt.grid(True)

    # Super title for the figure
    plt.suptitle('Comparison of RMSE Across Different Methods (Filtered)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("rmse_figure_2.png")
    # plt.show()

    print("Conventional UNet")

    indices = [0, 1, 2, 3, 4]

    fig, axes = plt.subplots(5, 5, figsize=(50, 15))
    fig.suptitle(
        '5 Representative Predicted Background, Ground Truth, Predicted Spectral, and Original Speimg Image Samples')

    sptimg4_test_notnorm = test_loader_notnorm.dataset.X
    tbg4_test_notnorm = test_loader_notnorm.dataset.Y

    tbg4_test_max, tbg4_test_min = np.max(tbg4_test_notnorm), np.min(tbg4_test_notnorm)
    sptimg4_test_max, sptimg4_test_min = np.max(sptimg4_test_notnorm), np.min(sptimg4_test_notnorm)


    # Loop to plot all graphs
    for i, idx in enumerate(indices):
        predicted_background = YPred[idx, :, :].transpose(1, 0)
        ground_truth = tbg4_test[idx, :, :].transpose(1, 0)
        original_speimg = sptimg4_test[idx, :, :].transpose(1, 0)
        # ground_truth_spectral_image = gt_spt_test[idx, :, :].transpose(1, 0)

        if config.NORMALIZATION and config.DENORMALIZATION:
            ground_truth = ground_truth * (tbg4_test_max - tbg4_test_min) + tbg4_test_min
            predicted_background = predicted_background * (sptimg4_test_max - sptimg4_test_min) + sptimg4_test_min
            original_speimg = original_speimg * (sptimg4_test_max - sptimg4_test_min) + sptimg4_test_min

        predicted_spectral_image = original_speimg - predicted_background
        ground_truth_spectral_image = original_speimg - ground_truth

        if config.NORMALIZATION and config.DENORMALIZATION:
            predicted_spectral_image = (predicted_spectral_image - np.min(predicted_spectral_image)) / \
                                       (np.max(predicted_spectral_image) - np.min(predicted_spectral_image) + 1e-8)
        if idx == 1:
            print("Predicted Background {}".format(idx))
            print("Minimum Intensity: {}".format(np.min(predicted_background)))
            print("Maximum Intensity: {}".format(np.max(predicted_background)))

            print("Ground Truth Background {}".format(idx))
            print("Minimum Intensity: {}".format(np.min(ground_truth)))
            print("Maximum Intensity: {}".format(np.max(ground_truth)))

            print("Original Spectral Image {}".format(idx))
            print("Minimum Intensity: {}".format(np.min(original_speimg)))
            print("Maximum Intensity: {}".format(np.max(original_speimg)))

        ax = axes[i, 0]
        ax.imshow(predicted_background, aspect='auto', cmap='gray')
        ax.set_title(f'Predicted Background {idx}')

        ax = axes[i, 1]
        ax.imshow(ground_truth, aspect='auto', cmap='gray')
        ax.set_title(f'Ground Truth Background {idx}')

        ax = axes[i, 2]
        ax.imshow(predicted_spectral_image, aspect='auto', cmap='gray')
        ax.set_title(f'Predicted Spectral Image {idx}')

        ax = axes[i, 3]
        ax.imshow(ground_truth_spectral_image, aspect='auto', cmap='gray')
        ax.set_title(f'Ground Truth Spectral Image {idx}')

        ax = axes[i, 4]
        ax.imshow(original_speimg, aspect='auto', cmap='gray')
        ax.set_title(f'Original Spectral Image {idx}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
    plt.savefig("unet/convUNet_5_rep_samples.png")

    print("Conventional UNet with Attention")

    fig, axes = plt.subplots(5, 5, figsize=(50, 15))
    fig.suptitle(
        '5 Representative Predicted Background, Ground Truth, Predicted Spectral, and Original Speimg Image Samples')

    # Loop to plot all graphs
    for i, idx in enumerate(indices):
        predicted_background = YPred2[idx, :, :].transpose(1, 0)
        ground_truth = tbg4_test[idx, :, :].transpose(1, 0)
        original_speimg = sptimg4_test[idx, :, :].transpose(1, 0)
        # ground_truth_spectral_image = gt_spt_test[idx, :, :].transpose(1, 0)

        if config.NORMALIZATION and config.DENORMALIZATION:
            ground_truth = ground_truth * (tbg4_test_max - tbg4_test_min) + tbg4_test_min
            predicted_background = predicted_background * (sptimg4_test_max - sptimg4_test_min) + sptimg4_test_min
            original_speimg = original_speimg * (sptimg4_test_max - sptimg4_test_min) + sptimg4_test_min

        predicted_spectral_image = original_speimg - predicted_background
        ground_truth_spectral_image = original_speimg - ground_truth

        if config.NORMALIZATION and config.DENORMALIZATION:
            predicted_spectral_image = (predicted_spectral_image - np.min(predicted_spectral_image)) / \
                                       (np.max(predicted_spectral_image) - np.min(predicted_spectral_image) + 1e-8)

        if idx == 1:
            print("Predicted Background {}".format(idx))
            print("Minimum Intensity: {}".format(np.min(predicted_background)))
            print("Maximum Intensity: {}".format(np.max(predicted_background)))

            print("Ground Truth Background {}".format(idx))
            print("Minimum Intensity: {}".format(np.min(ground_truth)))
            print("Maximum Intensity: {}".format(np.max(ground_truth)))

            print("Original Spectral Image {}".format(idx))
            print("Minimum Intensity: {}".format(np.min(original_speimg)))
            print("Maximum Intensity: {}".format(np.max(original_speimg)))

        ax = axes[i, 0]
        ax.imshow(predicted_background, aspect='auto', cmap='gray')
        ax.set_title(f'Predicted Background {idx}')

        ax = axes[i, 1]
        ax.imshow(ground_truth, aspect='auto', cmap='gray')
        ax.set_title(f'Ground Truth Background {idx}')

        ax = axes[i, 2]
        ax.imshow(predicted_spectral_image, aspect='auto', cmap='gray')
        ax.set_title(f'Predicted Spectral Image {idx}')

        ax = axes[i, 3]
        ax.imshow(ground_truth_spectral_image, aspect='auto', cmap='gray')
        ax.set_title(f'Ground Truth Spectral Image {idx}')

        ax = axes[i, 4]
        ax.imshow(original_speimg, aspect='auto', cmap='gray')
        ax.set_title(f'Original Spectral Image {idx}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
    plt.savefig("attention_unet/convUNet_Att_5_rep_samples.png")
    # plt.show()
