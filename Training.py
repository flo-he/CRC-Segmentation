import os

import torch
import torch.optim as optim
from torchvision.transforms import ToTensor, Normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import MirrorPad
from CRC_Dataset import CRC_Dataset
from trainer import Trainer
from multiprocessing import cpu_count

from utils import complex_net, Dice_Loss, Dice_and_CE

import logging
from U_Net import NeuralNet


# GLOBAL TRAINING PARAMETERS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = torch.cuda.is_available()

train_dict= {
    "device" : device,
    "epochs" : 250,
    "batch_size" : 1,
    "cv_folds": 5,
    "images_per_epoch" : 300,
    "pin_mem" : torch.cuda.is_available(),
    "num_workers" : 2,
    "output_dir" : "model_affine\\",
    "train_from_chkpts" : ["C:\\AML_seg_proj\\CRC-Segmentation\\model_affine\\model_chkpt_91.pt",
                           "C:\\AML_seg_proj\\CRC-Segmentation\\model_affine\\optimizer_chkpt_91.pt"],
    "log_level" : logging.INFO
}

def main():
    # create pytorch dataset
    dataset_tr = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), "data\\train"),
        transforms = [MirrorPad(((6,), (6,), (0,))), ToTensor(), Normalize(mean=(0.7979, 0.6772, 0.7768), std=(0.1997, 0.3007, 0.2039), inplace=True)]
    )

    # set model, optimizer and loss criterion
    model = NeuralNet(64, 128, 256, 512, 1024, droprate=0.33)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-5)
    criterion = Dice_and_CE(device).to(device)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, min_lr=1e-6, verbose=True)

    # initialize trainer class
    trainer = Trainer(model, optimizer, criterion, lr_scheduler, dataset_tr, **train_dict)

    # start training
    trainer.run_training()


if __name__ == "__main__":
    main()