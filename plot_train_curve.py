import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import os
import argparse

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--loss_array', type=str, metavar='', required=True, help='Filename of stored loss array.')
parser.add_argument('-i', '--identifier', type=str, metavar='', required=True, help='Outputfile identifier (folder of model used).')

args = parser.parse_args()

working_dir = os.getcwd()
try:
    os.mkdir(os.path.join(working_dir, "training_plots"))
except FileExistsError:
    pass

train_data_path = args.checkpoint_file

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
plt.savefig(os.path.join(working_dir, "training_plots", "loss_curve_" + args.identifier + ".png"))
plt.show()

# ema losses
plt.figure(figsize=(15, 8))
plt.plot(epoch_arr, train_data[:, 2], label="Train Loss EMA")
plt.plot(epoch_arr, train_data[:, 3], label="Validation Loss EMA")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "training_plots", "EMA_loss_curve_" + args.identifier + ".png"))
plt.show()
