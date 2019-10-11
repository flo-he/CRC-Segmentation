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
    # ~ 50.000 batch updates -> 50.000/200 = 250 epochs
    "epochs" : 250,
    "batch_size" : 16,
    "cv_folds": 5,
    # 1 epoch = 250 batches -> images per epoch = batch size * batches per epoch
    "images_per_epoch" : int(250*16),
    "pin_mem" : torch.cuda.is_available(),
    "num_workers" : 2,
    "output_dir" : "model_large_drop_batch_ce\\",
    "train_from_chkpts" : ["C:\AML_seg_proj\CRC-Segmentation\model_large_drop_batch_ce\model_chkpt_65.pt",
                           "C:\AML_seg_proj\CRC-Segmentation\model_large_drop_batch_ce\optimizer_chkpt_65.pt",
                           "C:\AML_seg_proj\CRC-Segmentation\model_large_drop_batch_ce\loss_arr_65.pt"],
    "log_level" : logging.DEBUG
}

def main():
    # create pytorch dataset
    dataset_tr = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), "data\\train"),
        transforms = [
            transforms.RandomCrop(crop_shape=(256, 256)),
            transforms.RandomFlip(),
            #transforms.MirrorPad(padding=((3,), (3,), (0,))),
            transforms.ToTensor(),
            transforms.Normalize(means=(0.7942, 0.6693, 0.7722), stds=(0.1998, 0.3008, 0.2037))
        ]
    )

    # set model, optimizer and loss criterion
    model = UNet((256, 256), (256, 256), 32, 64, 128, 256, 512, droprate=0.5, Norm=nn.BatchNorm2d)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-5)

    # compute class weights
    #t = 1/torch.tensor([131383/250000, 68638/250000, 49979/250000])
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5, min_lr=1e-6, verbose=True)

    # initialize trainer class
    trainer = Trainer(model, optimizer, criterion, lr_scheduler, dataset_tr, **train_dict)
    
    # start training
    trainer.run_training()


if __name__ == "__main__":
    main()