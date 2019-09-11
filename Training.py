import os

import torch
from torchvision.transforms import ToTensor

from transformations import MirrorPad
from CRC_Dataset import CRC_Dataset
from trainer import Trainer
from multiprocessing import cpu_count

# GLOBAL TRAINING PARAMETERS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(device=device)} for training.")

train_dict= {
    "device" : device,
    "epochs" : 100,
    "batch_size" : 32,
    "cv_folds": 5,
    "pin_mem" : torch.cuda.is_available(),
    "num_workers " : cpu_count()
}


def main():

    # create pytorch dataset
    dataset_tr = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), "data\\train"),
        transforms = [MirrorPad(((6,), (6,), (0,))), ToTensor()]
    )

    # set model, optimizer and loss criterion
    model = ...
    optimizer = ...
    criterion = ...

    # initialize trainer class
    trainer = Trainer(model, optimizer, criterion, dataset_tr, **train_dict)

    # start training
    trainer.run_training()


if __name__ == "__main__":
    main()