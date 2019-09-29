import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import os

model_folder = "model_test"
working_dir = os.getcwd()
try:
    os.mkdir(os.path.join(working_dir, "training_plots"))
except FileExistsError:
    pass

train_data_path = os.path.join(working_dir, model_folder, "")

# load in data
train_data = torch.load(train_data_path).numpy()
print(train_data.shape)
n_epochs = train_data.shape[0] + 1
epoch_arr = np.arange(n_epochs)[1:]

# losses
plt.figure(figsize=(15, 8))
plt.plot(epoch_arr, train_data[:, 0], label="Train Loss")
plt.plot(epoch_arr, train_data[:, 1], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "training_plots", "loss_curve_" + model_folder + ".png"))
plt.show()

# ema losses
plt.figure(figsize=(15, 8))
plt.plot(epoch_arr, train_data[:, 2], label="Train Loss EMA")
plt.plot(epoch_arr, train_data[:, 3], label="Validation Loss EMA")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "training_plots", "EMA_loss_curve_" + model_folder + ".png"))
plt.show()
