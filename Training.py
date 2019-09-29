import os

import torch
import torch.optim as optim
import torch.nn as nn

from CRC_Dataset import CRC_Dataset
from trainer import Trainer

from utils import Dice_Loss, Dice_and_CE

import transforms
import logging

from U_Net import UNet


# GLOBAL TRAINING PARAMETERS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = torch.cuda.is_available()

train_dict= {
    "device" : device,
    # ~ 55.000 batch updates -> 55.000/250 = 220 epochs
    "epochs" : 220,
    "batch_size" : 16,
    "cv_folds": 5,
    # 1 epoch = 250 batches -> images per epoch = batch size * batches per epoch
    "images_per_epoch" : int(250*16),
    "pin_mem" : torch.cuda.is_available(),
    "num_workers" : 2,
    "output_dir" : "model_test\\",
    "train_from_chkpts" : [],
    "log_level" : logging.DEBUG
}

def main():
    # create pytorch dataset
    dataset_tr = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), "data\\train"),
        transforms = [
            transforms.RandomCrop(crop_shape=(250, 250)),
            transforms.RandomFlip(),
            transforms.MirrorPad(padding=((3,), (3,), (0,))),
            transforms.ToTensor(),
            transforms.Normalize(means=(0.7979, 0.6772, 0.7768), stds=(0.1997, 0.3007, 0.2039))
        ]
    )

    # set model, optimizer and loss criterion
    model = UNet((256, 256), (250, 250), 64, 128, 256, 512)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-5)
    # use reweighted cross entropy
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1/131383, 1/68638, 1/49979]))
    #criterion = Dice_and_CE(device).to(device)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, min_lr=1e-6, verbose=True)

    # initialize trainer class
    trainer = Trainer(model, optimizer, criterion, lr_scheduler, dataset_tr, **train_dict)

    # start training
    trainer.run_training()


if __name__ == "__main__":
    main()