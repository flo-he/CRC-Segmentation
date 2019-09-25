import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import logging
import os

class Dice_Loss(nn.Module):
    '''
    Implements dice loss for a multi class problem.

    Args: \n
        :n_classes: int, number of classes in the target mask   \n
    Calling:    \n
        :pred: output of the segmentation network of shape [B, C, H, W]
        :target: target segmentation mask of shape [B, H, W]
    '''
    
    def __init__(self, device, n_classes=3):
        super(Dice_Loss, self).__init__()
        self.n_classes = n_classes
        self.device = device

        if self.device.type == "cuda":
            self.tensortype = torch.cuda.FloatTensor
        else:
            self.tensortype = torch.FloatTensor

    def forward(self, pred, target):
        
        # convert target N x H x W to N x n_classes x H x W
        target_one_hot = torch.eye(self.n_classes)[target].permute(0, 3, 1, 2).type(self.tensortype)

        # compute softmax of prediction
        probs = nn.functional.softmax(pred, dim=1)

        nominator = torch.sum(probs * target_one_hot, (2, 3))
        denominator = torch.sum(probs + target_one_hot, (2, 3))

        # negated loss, since pytorch optimizer minimizes loss
        dice_per_inst = -2. * torch.mean(nominator/denominator, 1)
        
        # mean over batch
        return  dice_per_inst.mean()
        
class Dice_Score(nn.Module):
    def __init__(self, device, n_classes=3):
        super(Dice_Score, self).__init__()
        self.dice_loss = Dice_Loss(device, n_classes)

    def forward(self, pred, target):
        return -self.dice_loss(pred, target)

class Dice_and_CE(nn.Module):
    '''
    Implements both Dice and cross entropy loss added.  \n
    See Dice_loss doc and pytorch's CrossEntropyLoss class for info.
    '''

    def __init__(self, device,  n_classes=3):
        super(Dice_and_CE, self).__init__()
        self.n_classes = n_classes
        self.dice = Dice_Loss(device, n_classes)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.ce(pred, target)

class MirrorPad(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, image):
        return np.pad(image, self.padding, mode='symmetric')

class complex_net(nn.Module):
    '''
    Literally just for testing the trainer.
    '''
    def __init__(self):
        super(complex_net, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.activ = nn.ReLU()

    def forward(self, x):
        return self.activ(self.conv(self.conv(self.conv(x))))

class Pixel_Accuracy(nn.Module):
    def __init__(self, device):
        super(Pixel_Accuracy, self).__init__()
        self.mask_shapes = (500, 500)
        self.n_pixel = np.prod(self.mask_shapes)
        self.device = device 

        if self.device.type == "cuda":
            self.tensortype = torch.cuda.FloatTensor
        else:
            self.tensortype = torch.FloatTensor


    def forward(self, pred, target):
        # convert network prediction to 2d mask
        probs = nn.functional.softmax(pred, dim=1)
        pred_masks = torch.argmax(probs, dim=1)

        # flatten image dims and compute difference between images
        diff = (pred_masks - target).view(-1, self.n_pixel)

        # count correctly predicted pixels per batch instance
        accuracies = (diff == 0).sum(dim=1).type(self.tensortype) / self.n_pixel

        return accuracies.mean()

class Loss_EMA(nn.Module):
    def __init__(self, alpha=0.9):
        super(Loss_EMA, self).__init__()
        self.alpha = alpha
        # track both validation and training loss averages
        self.validation_avg = []
        self.training_avg = []

    def forward(self, tr, val):

        # first time step
        if not self.validation_avg:
            new_ma_tr, new_ma_val = tr, val
            self.training_avg.append(new_ma_tr)
            self.validation_avg.append(new_ma_val)
        else:   
            # update rule of moving averages
            new_ma_tr = (1. - self.alpha) * tr + self.alpha * self.training_avg[-1]
            new_ma_val = (1. - self.alpha) * val + self.alpha * self.validation_avg[-1]
            self.training_avg.append(new_ma_tr)
            self.validation_avg.append(new_ma_val)
        
        return new_ma_tr, new_ma_val


def get_train_logger(log_level):
    logger = logging.getLogger("TRAINER")
    now = datetime.now()
    date_str = now.strftime("%d_%m_%Y_%H.%M.%S")

    # create log directory if it does not exist
    try:
        os.mkdir(".\\log")
    except FileExistsError:
        pass
    
    logging.basicConfig(level=log_level)
    fh = logging.FileHandler(f"log\\train_{date_str}.log")
    formatter = logging.Formatter('%(asctime)s | %(name)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
