import torch
from torchvision.transforms import ToTensor, Compose
import numpy as np
import glob
import os
import time 
from transformations import MirrorPad
import matplotlib.pyplot as plt
    

class CRC_Dataset(torch.utils.data.Dataset):
    '''
    CRC Tissue Segmentation Dataset. Files are assumed to lie in a root directory with subfolders 'frames' and 'masks' with the compressed .npz file format.
    (May be changed once considering split between training and test set and if loading compressed .npz files becomes a bottleneck in dataloading & preprocessing).
    Further the images are assumed to be in a RGB format and unprocessed.
    '''

    def __init__(self, root_dir, transforms=[ToTensor()]):
        '''
        Args:\n
            :root_dir: string containing the root directory of the dataset. 
            :transforms: list of transformations to apply to the samples.
        '''
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.composed_trsfm = Compose(transforms)

        # save name of files in lists
        self.images = glob.glob(os.path.join(root_dir, 'frames\\*.npz'))
        self.labels = glob.glob(os.path.join(root_dir, 'masks\\*.npz'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Loads an image and its segmentation mask from disc, applies transformations and returns them as pytorch tensors.
        '''
        
        # load image and label from disk
        image = np.load(self.images[idx])['arr_0']
        label = np.load(self.labels[idx])['arr_0']

        # apply transformations 
        image = self.composed_trsfm(image)

        return image, torch.from_numpy(label).long()


if __name__ == "__main__":

    # test
    dataset = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), 'data\\train'),
        transforms = [MirrorPad(((92,), (92,), (0,))), ToTensor()]
    )

    print(dataset.images[:10], dataset.labels[:10], len(dataset))

    t1 = time.perf_counter()
    image, label = dataset[np.random.randint(len(dataset))]
    t2 = time.perf_counter() - t1

    print(f"{CRC_Dataset.__getitem__} took {t2}s")
    print(image.shape, label.shape, image.dtype, label.dtype)

    plt.imshow(image.numpy().transpose((1, 2, 0)))
    plt.show()

   








