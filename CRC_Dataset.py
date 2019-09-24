import torch
from torchvision.transforms import ToTensor, Compose
import numpy as np
import glob
import os
#import time 
#from utils import MirrorPad
#import matplotlib.pyplot as plt
    
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

def compute_mean_and_std(dataloader, device):
    '''
    Compute the mean and standard deviation per channel of the whole dataset using first and second moments.
    '''

    # running means of first and sec. moment
    mean = torch.zeros(3).to(device)
    mean_sq = torch.zeros(3).to(device)

    # loop over training set in batches, compute pixel means and batch means 
    for img_batch, _ in dataloader:
        n_img, channel = img_batch.size(0), img_batch.size(1)
        img_batch = img_batch.view(n_img, channel, -1).to(device)

        mean += torch.mean(img_batch, (0, 2))
        mean_sq += torch.mean(img_batch**2, (0, 2))

    # divide by # of batches 
    mean /= len(dataloader)
    mean_sq /= len(dataloader)

    # return means and stds
    return mean.cpu(), torch.sqrt(mean_sq - mean**2).cpu()


    
if __name__ == "__main__":
    # test
    dataset = CRC_Dataset(root_dir = os.path.join(os.getcwd(), 'data\\train'))

    #print(dataset.images[:10], dataset.labels[:10], len(dataset))

    #t1 = time.perf_counter()
    #image, label = dataset[np.random.randint(len(dataset))]
    #t2 = time.perf_counter() - t1

    #print(f"{CRC_Dataset.__getitem__} took {t2}s")
    #print(image.shape, label.shape, image.dtype, label.dtype)

    #plt.imshow(image.numpy().transpose((1, 2, 0)))
    #plt.show()

    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 32,
        pin_memory = True,
        num_workers = 2
    )

    mean, std = compute_mean_and_std(dataloader, device="cuda:0") #set to "cpu" if not gpu is available
    print(mean, std) #prints approx. mean=(0.7979, 0.6772, 0.7768), std=(0.1997, 0.3007, 0.2039)




   








