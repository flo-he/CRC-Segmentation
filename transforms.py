import torch 
import numpy as np
import os
import matplotlib.pyplot as plt

from CRC_Dataset import CRC_Dataset

class RandomCrop(object):
    '''
    Implements Random Crop for both squared input images and masks.

    Args: \n
        :crop_shape: tuple, output image/mask shape
    '''

    def __init__(self, crop_shape):
        assert crop_shape[0] == crop_shape[1]
        self.crop_shape = crop_shape
        self.crop_along_dim = crop_shape[0]

    def __call__(self, sample):
        img, mask = sample
        img_h, mask_h = img.shape[0], mask.shape[0]
        
        available_range = (img_h - self.crop_along_dim) // 2

        offset = np.random.randint(available_range)

        img_cropped = img[offset:offset+self.crop_along_dim, offset:offset+self.crop_along_dim, :]
        mask_cropped = mask[offset:offset+self.crop_along_dim, offset:offset+self.crop_along_dim]
        
        return (img_cropped, mask_cropped)

class MirrorPad(object):
    '''
    Mirrors the image in symmetric mode, keeps mask untouched

    Args:
        :padding: tuple, padding shapes per dim
    '''

    def __init__(self, padding):
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample
        return (np.pad(img, self.padding, mode='symmetric'), mask)

class RandomFlip(object):
    '''
    Flips image and mask with a 25% chance horizontally or vertically and keeps them untouched with 50% chance.
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        img, mask = sample

        if np.random.uniform() > 0.5:
            # apply flip
            if np.random.uniform() > 0.5:
                # flip horizontally
                img, mask = img[:, ::-1, :], mask[:, ::-1]
            else:
                # flip vertically
                img, mask = img[::-1, :, :], mask[::-1, :]
        
        return (np.ascontiguousarray(img), np.ascontiguousarray(mask))

class ToTensor(object):
    '''
    Does practically the same as torchvision.transforms.ToTensor but additionally casts the mask to Pytorch long tensor.
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        img, mask = sample

        # map image into the interval [0, 1]
        img = img.astype(np.float32) / 255.

        # transpose [H, W, C] -> [C, H, W]
        img = np.transpose(img, (2, 0, 1))

        return (torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(mask)).long())

class Normalize(object):

    '''
    Does practically the same as torchvision.transforms.Normalize but passes mask through. Expects Pytorch tensors of shape (C, H, W).
    Args:
        :means: tuple or list, per channel mean
        :stds: tuple or list, per channel standard deviation
    '''

    def __init__(self, means, stds):
        self.means = torch.tensor(means).view(-1, 1, 1)
        self.stds = torch.tensor(stds).view(-1, 1, 1)

    def __call__(self, sample):
        img, mask = sample

        img = (img - self.means) / self.stds

        return (img, mask)


if __name__ == "__main__":

    dataset = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), 'data\\train'),
    )

    # transforms
    crop = RandomCrop((250, 250))
    flip = RandomFlip()
    totensor = ToTensor()
    mirror = MirrorPad(padding=((3,), (3,), (0,)))
    normalize = Normalize(means=(0.7979, 0.6772, 0.7768), stds=(0.1997, 0.3007, 0.2039))

    i = np.random.randint(len(dataset))

    img, mask = dataset[i]

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask, cmap="gray")
    plt.show()

    img, mask = crop((img, mask))

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask, cmap="gray")
    plt.show()

    img, mask = flip((img, mask))

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask, cmap="gray")
    plt.show()

    img, mask = mirror((img, mask))
    
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask, cmap="gray")
    plt.show()

    img, mask = totensor((img, mask))
    print(img.size(), img.type(), mask.size(), mask.type())

    img, mask = normalize((img, mask))


