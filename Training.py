import os

import torch
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import MirrorPad
from CRC_Dataset import CRC_Dataset
from trainer import Trainer
from multiprocessing import cpu_count

from utils import complex_net, Dice_Loss, Dice_and_CE

import logging


# GLOBAL TRAINING PARAMETERS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dict= {
    "device" : device,
    "epochs" : 100,
    "batch_size" : 32,
    "cv_folds": 3,
    "images_per_epoch" : 1000,
    "pin_mem" : torch.cuda.is_available(),
    "num_workers" : 2,
    "output_dir" : "models\\",
    "train_from_chkpts" : [],#["C:\AML_seg_proj\CRC-Segmentation\models\model_chkpt_25.pt", "C:\AML_seg_proj\CRC-Segmentation\models\optimizer_chkpt_25.pt"],
    "log_level" : logging.DEBUG
}

def main():
    # create pytorch dataset
    dataset_tr = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), "data\\train"),
        transforms = [MirrorPad(((6,), (6,), (0,))), ToTensor()]
    )

    # set model, optimizer and loss criterion
    model = complex_net()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = Dice_and_CE(device).to(device)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True)

    # initialize trainer class
    trainer = Trainer(model, optimizer, criterion, lr_scheduler, dataset_tr, **train_dict)

    # start training
    trainer.run_training()


if __name__ == "__main__":
    main()