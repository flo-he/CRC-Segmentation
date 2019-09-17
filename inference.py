import numpy as np
import torch
from torchvision.transforms import ToTensor
from utils import MirrorPad, complex_net, Dice_Score, Pixel_Accuracy
from multiprocessing import cpu_count
from CRC_Dataset import CRC_Dataset
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
chkpt = "C:\AML_seg_proj\CRC-Segmentation\models\model_chkpt_5.pt"

def compute_metrics_on_test_set(model, test_loader):

    n_batches = len(test_loader)

    # perfomance evaluation metrics
    dice = Dice_Score(device).to(device)
    px_acc = Pixel_Accuracy(device).to(device)


    dice_sc = 0.
    acc = 0.

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

    print(dice_sc/n_batches, acc/n_batches)


def main():

    # get test set and test set loader
    test_set = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), "data\\test"),
        transforms = [MirrorPad(((6,), (6,), (0,))), ToTensor()]
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size = 32,
        num_workers = cpu_count(),
        pin_memory = use_cuda,        
    )

    model = complex_net()
    model.to(device)
    
    # load model checkpoint
    model.load_state_dict(torch.load(chkpt))

    compute_metrics_on_test_set(model, test_loader)

    



if __name__ == "__main__":
    main()


