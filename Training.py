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


# GLOBAL TRAINING PARAMETERS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dict= {
    "device" : device,
    "epochs" : 100,
    "batch_size" : 32,
    "cv_folds": 5,
    "pin_mem" : torch.cuda.is_available(),
    "num_workers" : cpu_count(),
    "output_dir" : "models\\",
    "train_from_chkpts" : []
}

def main():
    if device.type == "cuda":
        print(f"Using {torch.cuda.get_device_name(device=device)} for training.")
    else:
        print("Using CPU for training.")

    # create pytorch dataset
    dataset_tr = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), "data\\train"),
        transforms = [MirrorPad(((6,), (6,), (0,))), ToTensor()]
    )

    # set model, optimizer and loss criterion
    model = complex_net()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = Dice_and_CE(device).to(device)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, min_lr=1e-6)

    # initialize trainer class
    trainer = Trainer(model, optimizer, criterion, lr_scheduler, dataset_tr, **train_dict)

    # start training
    trainer.run_training()


if __name__ == "__main__":
    main()