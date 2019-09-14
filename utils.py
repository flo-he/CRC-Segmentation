import torch
import torch.nn as nn
import numpy as np

class Dice_Loss(nn.Module):
    '''
    Implements dice loss for a multi class problem with optional smoothing.

    Args: \n
        :n_classes: int, number of classes in the target mask   \n
        :smoothing: bool, if true, smoothing is applied (avoid division by zero)    \n
    Calling:    \n
        :pred: output of the segmentation network of shape [B, C, H, W]
        :target: target segmentation mask of shape [B, H, W]
    '''
    
    def __init__(self, device, n_classes=3, smoothing=True):
        super(Dice_Loss, self).__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.device = device

    def forward(self, pred, target):
        
        # convert target N x H x W to N x n_classes x H x W
        if self.device.type == "cuda":
            target_one_hot = torch.eye(self.n_classes)[target].permute(0, 3, 1, 2).cuda()
        else:
            target_one_hot = torch.eye(self.n_classes)[target].permute(0, 3, 1, 2).float()

        # compute softmax of prediction
        probs = nn.functional.softmax(pred, dim=1)

        if self.smoothing:
            # torch.sum: sum tensors along all dimensions but batch dimension
            numerator = 2. * torch.sum(probs * target_one_hot, (1, 2, 3)) + 1
            denominator = torch.sum(probs + target_one_hot, (1, 2, 3)) + 1
        else:
            numerator = 2. * torch.sum(probs * target_one_hot, (1, 2, 3))
            denominator = torch.sum(probs + target_one_hot, (1, 2, 3))

        # return negated loss, since pytorch optimizer minimizes loss
        return 1 - (numerator / denominator).mean() # mean over batch

class Dice_and_CE(nn.Module):
    '''
    Implements both Dice and cross entropy loss added.  \n
    See Dice_loss doc and pytorch's CrossEntropyLoss class for info.
    '''

    def __init__(self, device,  n_classes=3, smoothing=True):
        super(Dice_and_CE, self).__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.dice = Dice_Loss(device, n_classes, smoothing)
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