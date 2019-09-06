import torch
from torchvision.transforms import Compose
import numpy as np
import glob
import os
import time 

class CRC_Dataset(torch.utils.data.Dataset):
    '''
    CRC Tissue Segmentation Dataset. Files are assumed to lie in a root directory with subfolders 'frames' and 'masks' with the .npy file format.
    (May be changed once considering split between training and test set).
    Further the images are assumed to be in a RGB format and unprocessed.
    '''

    def __init__(self, root_dir, transforms=None):
        '''
        Args:\n
            :root_dir: string containing the root directory of the dataset. 
            :transforms: list of transformations to apply to the samples.
        '''
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.composed_trsfm = Compose(transforms)

        # save name of files in lists
        self.images = glob.glob(os.path.join(root_dir, 'frames\\*.npy'))
        self.labels = glob.glob(os.path.join(root_dir, 'masks\\*.npy'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Loads an image and its segmentation mask from disc and returns them as pytorch tensors.
        '''
        
        # load image and label from disk
        image = np.load(self.images[idx])
        label = np.load(self.labels[idx])

        #print(image.shape, image.dtype)

        # convert them to pytorch tensors, specifically, adjust image spacing for pytorch images (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        label = torch.from_numpy(label)

        # apply transformatons if wanted
        if self.transforms:
            image, label = self.composed_trsfm((image, label))

        return image, label


if __name__ == "__main__":

    # test
    dataset = CRC_Dataset(os.path.join(os.getcwd(), 'data'))

    print(dataset.images[:10], dataset.labels[:10], len(dataset))

    t1 = time.perf_counter()
    image, label = dataset[0]
    t2 = time.perf_counter() - t1

    print(f"{CRC_Dataset.__getitem__} took {t2}s")
    print(image.shape, label.shape, image.dtype, label.dtype)









