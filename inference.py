import torch
from transforms import MirrorPad, ToTensor, Normalize
from utils import Dice_Score, Pixel_Accuracy
from CRC_Dataset import CRC_Dataset
from U_Net import UNet
import os

import argparse

# Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint_file', type=str, metavar='', required=True, help='Path to the file of the model state_dict to use')
parser.add_argument('-w', '--workers', type=int, metavar='', help='Number of dataloading workers. Default: 1')
parser.add_argument('-b', '--batch_size', type=int, metavar='', help='Batch size to use. Default: 8')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
chkpt = args.checkpoint_file

def compute_metrics_on_test_set(model, test_loader):

    n_batches = len(test_loader)

    # perfomance evaluation metrics
    dice = Dice_Score()
    px_acc = Pixel_Accuracy((500, 500))


    dice_sc = 0.
    acc = 0.

    model.eval()
    with torch.no_grad():
        for idx, (image, mask) in enumerate(test_loader):
            # get image batch and label batch
            image = image.to(device, non_blocking=use_cuda)
            mask = mask.to(device, non_blocking=use_cuda)

            # model output
            output = model(image)
            del image

            # metrics
            dice_sc += dice(output, mask)
            acc += px_acc(output, mask)
            del mask

    return dice_sc/n_batches, acc/n_batches


def main():

    # get test set and test set loader
    test_set = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), "data\\test"),
        transforms = [MirrorPad(((6,), (6,), (0,))), ToTensor(), Normalize(means=(0.7942, 0.6693, 0.7722), stds=(0.1998, 0.3008, 0.2037))]
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size = args.batch_size if args.batch_size else 8,
        num_workers = args.workers if args.workers else 1,
        pin_memory = use_cuda,        
    )

    model = UNet((512, 512), (500, 500), 64, 128, 256, 512)
    model.load_state_dict(torch.load(chkpt))
    model.to(device)
    
    compute_metrics_on_test_set(model, test_loader)



if __name__ == "__main__":
    main()


