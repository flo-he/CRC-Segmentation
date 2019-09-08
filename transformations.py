import numpy as np
import torch

class MirrorPad(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, image):
        return np.pad(image, self.padding, mode='symmetric')
    