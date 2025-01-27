import config
import torch
from utils import get_model_path_filename
import numpy as np
import matplotlib.pyplot as plt

def save_loss_graphs(model_name, training_loss_epochs, testing_loss_epochs):
    x_labels = np.arange(1, config.EPOCHS)
    training_losses = training_loss_epochs[1:]  # Ignore 1st epoch
    testing_losses = testing_loss_epochs[1:]
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, training_losses, label="Training RMSE", color="b", marker="o")
    plt.plot(x_labels, testing_losses, label="Testing Data", color="r", marker="o")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.legend()

    if model_name == 'unet':
        model_name = "conv_UNet"
    elif model_name == 'attention_unet':
        model_name = "conv_att_UNet"

    plt.savefig(f"{model_name}_training_testing_rmse_graph.png")

    np.save(f'{model_name}_training_losses.npy', np.array(training_losses))
    np.save(f'{model_name}_testing_losses.npy', np.array(testing_losses))


def train(model_name, model, train_loader, test_loader, criterion, optimizer, scheduler):

    training_loss_epochs = []
    testing_loss_epochs = []

    # Training loop
    for epoch in range(config.EPOCHS):
        model.train()
        training_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)

        training_loss_epoch = training_loss / len(train_loader.dataset)

        print(f"Epoch [{epoch + 1}/{config.EPOCHS}], Training Loss: {training_loss_epoch}")

        model.eval()
        testing_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(inputs)
                testing_loss += criterion(outputs, targets).item() * inputs.size(0)

        testing_loss_epoch = testing_loss / len(test_loader.dataset)
        print(f"Epoch [{epoch + 1}/{config.EPOCHS}], Testing Loss: {testing_loss_epoch}")

        model.train()
        scheduler.step()

        training_loss_epochs.append(training_loss_epoch)
        testing_loss_epochs.append(testing_loss_epoch)

    torch.save(model.state_dict(), get_model_path_filename(model_name))

    save_loss_graphs(model_name, training_loss_epochs, testing_loss_epochs)

    print("Training complete!")

    return model