import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import config

def prepare_arrays(test_loader):
    sptimg4_test = test_loader.dataset.X
    GTspt_test = test_loader.dataset.Y

    # Determine number of test samples
    numSpectra = len(test_loader.dataset)  # Get total dataset length

    # Preallocate arrays for predictions
    YPred = np.zeros((numSpectra, 1, 128, 16))
    YPred2 = np.zeros((numSpectra, 1, 128, 16))

    return numSpectra, sptimg4_test, GTspt_test, YPred, YPred2

def save_rmse_figures(dataSets):
    # First figure
    plt.figure(figsize=(15, 20))

    # Subplot titles
    titles = [
        'Conv. Att. U-Net Spectra RMSE',
        'Conventional U-net Spectra RMSE',
        'Old Method Spectra RMSE'
    ]

    # Create subplots for each RMSE
    for i, rmse_data in enumerate(
            dataSets):
        plt.subplot(3, 1, i + 1)
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
    plt.figure(figsize=(15, 20))

    filteredDataSets = [data[data <= 15] for data in dataSets]  # Filter values greater than 15

    maxValue = max([data.max() for data in filteredDataSets if len(data) > 0], default=1)

    # Define global bin edges for the range 0-15
    buffer = 0
    binEdges = np.linspace(0, (1 + buffer) * maxValue, 31)

    # if not config.NORMALIZATION:
    #     binEdges = np.linspace(0, 15, 31)  # 30 bins from 0 to 15
    # else:
    #     binEdges = np.linspace(0, 1, 31)

    precision = ".5f" if config.NORMALIZATION else ".2f"

    # Loop through each filtered dataset to create subplot histograms with mean and std annotations
    for i, filtered_data in enumerate(filteredDataSets):
        plt.subplot(3, 1, i + 1)  # Set subplot position
        plt.hist(filtered_data, bins=binEdges, color='green', alpha=0.7)  # Plot histogram with defined bin edges

        meanVal = np.mean(filtered_data)
        stdVal = np.std(filtered_data)

        # Mean and standard deviation lines
        plt.axvline(meanVal, color='red', linestyle='-', linewidth=2, label=f'Mean: {meanVal:{precision}}')
        plt.axvline(meanVal + stdVal, color='blue', linestyle='--', linewidth=2,
                    label=f'+1 Std: {meanVal + stdVal:{precision}}')
        plt.axvline(meanVal - stdVal, color='blue', linestyle='--', linewidth=2,
                    label=f'-1 Std: {meanVal - stdVal:{precision}}')

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


def save_rep_samples(model_name, YPred, GTspt_test, sptimg4_test):
    indices = [0, 1, 2, 3, 4]

    fig, axes = plt.subplots(5, 3, figsize=(25, 15))
    if model_name == "unet":
        fig.suptitle(
            'Conventional UNet: 5 Representative Predicted Background, Ground Truth, Predicted Spectral, and Original Speimg Image Samples',
            fontsize=48, fontweight='bold')
    elif model_name == "attention_unet":
        fig.suptitle(
            'Attention UNet: 5 Representative Predicted Background, Ground Truth, Predicted Spectral, and Original Speimg Image Samples',
            fontsize=48, fontweight='bold')

    # Loop to plot all graphs
    for i, idx in enumerate(indices):
        predicted_spectral_image = YPred[idx, :, :].transpose(1, 0)
        ground_truth_spectral_image = GTspt_test[idx, :, :].transpose(1, 0)
        original_spectral_image = sptimg4_test[idx, :, :].transpose(1, 0)

        if idx == 1:
            print("Predicted Spectral Image {}".format(idx))
            print("Minimum Intensity: {}".format(np.min(predicted_spectral_image)))
            print("Maximum Intensity: {}".format(np.max(predicted_spectral_image)))

            print("Ground Truth Spectral Image {}".format(idx))
            print("Minimum Intensity: {}".format(np.min(ground_truth_spectral_image)))
            print("Maximum Intensity: {}".format(np.max(ground_truth_spectral_image)))

        ax = axes[i, 0]
        if config.NORMALIZATION:
            ax.imshow(predicted_spectral_image, aspect='auto', cmap='gray')
        else:
            ax.imshow(predicted_spectral_image, aspect='auto', cmap='gray')
        ax.set_title(f'Predicted Spectral Image {idx}', fontsize=32)

        ax = axes[i, 1]
        if config.NORMALIZATION:
            ax.imshow(ground_truth_spectral_image, aspect='auto', cmap='gray')
        else:
            ax.imshow(ground_truth_spectral_image, aspect='auto', cmap='gray')
        ax.set_title(f'Ground Truth Spectral Image {idx}', fontsize=32)

        ax = axes[i, 2]
        ax.imshow(original_spectral_image, aspect='auto', cmap='gray')
        ax.set_title(f'Original Spectral Image {idx}', fontsize=32)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
    if model_name == "unet":
        plt.savefig("unet/convUNet_5_rep_samples.png")
    elif model_name == "attention_unet":
        plt.savefig("attention_unet/convUNet_Att_5_rep_samples.png")


def test(models, test_loader):
    """Evaluates both models on the test dataset and stores results."""

    (numSpectra, sptimg4_test, GTspt_test, YPred, YPred2) = prepare_arrays(test_loader)

    # Set models to evaluation mode
    for model in models.values():
        model.eval()

    device = config.DEVICE

    inference_time_unet = 0
    inference_time_unet_att = 0

    for n in range(numSpectra):
        input = torch.tensor(sptimg4_test[n:n + 1], dtype=torch.float32).to(device)

        start_time = time.time()
        conv_UNet_output = models["unet"](input).detach().cpu().numpy()
        end_time = time.time()
        inference_time_unet += end_time - start_time
        YPred[n, :, :, :] = conv_UNet_output

        start_time = time.time()
        conv_att_UNet_output = models["attention_unet"](input).detach().cpu().numpy()
        end_time = time.time()
        inference_time_unet_att += end_time - start_time
        YPred2[n, :, :, :] = conv_att_UNet_output

    print("Average inference time (in seconds): ", {'unet': inference_time_unet / numSpectra, 'attention_unet': inference_time_unet_att / numSpectra})
    YPred = np.squeeze(YPred, axis=1)
    YPred2 = np.squeeze(YPred2, axis=1)

    sptimg4_test = np.squeeze(sptimg4_test, axis=1)

    GTspt_test = np.squeeze(GTspt_test, axis=1)

    def rmse(predictions, targets):
        """Calculate the RMSE between predictions and targets."""
        return np.sqrt(np.mean((predictions - targets) ** 2, axis=(1, 2)))

    ConvU_SPE_RMSE = rmse(YPred, GTspt_test)
    ConvU_Att_SPE_RMSE = rmse(YPred2, GTspt_test)

    # Plotting figures

    dataSets = [ConvU_Att_SPE_RMSE, ConvU_SPE_RMSE]
    save_rmse_figures(dataSets)

    print("Conventional UNet\n")
    save_rep_samples("unet", YPred, GTspt_test, sptimg4_test)

    print("Conventional UNet with Attention\n")
    save_rep_samples("attention_unet", YPred2, GTspt_test, sptimg4_test)
