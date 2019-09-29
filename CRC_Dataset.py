import torch
from torchvision.transforms import Compose
from transforms import ToTensor
import numpy as np
import glob
import os
#import time 
#from utils import MirrorPad
#import matplotlib.pyplot as plt

class CV_Splits(object):
    '''
    Generator for cross validation compatible with PyTorch Datasets. \n
    Args:   \n
        :cv_folds: number k of k-fold cross validation \n
        :shuffle: bool, shuffle indices before creating splits \n
        :dataset: initialize Generator with a PyTorch Dataset
    '''

    def __init__(self,  cv_folds, subset_size=None, shuffle=True, dataset=None):
        self.folds = cv_folds
        self.dataset = dataset
        self.shuffle = shuffle

        # determine how much of the available training data is used for each cross validation iteration
        self.subset_size = subset_size
        if subset_size:
            self.n_train = int((cv_folds - 1)/cv_folds * subset_size)
            self.n_val = int(subset_size/cv_folds)

    def __call__(self, dataset):
        '''
        Args:
            :dataset: PyTorch dataset to reassign
        '''
        self.dataset = dataset

    def __iter__(self):
        '''
        Generator method, yields training and validation set as PyTorch Datasets.
        '''
        self.indices = np.arange(len(self.dataset))

        if self.shuffle:
            self.indices = np.random.permutation(self.indices)

        #print(len(self.indices))
        # create splits
        self.splits = np.array_split(self.indices, self.folds)
        #print(self.splits)

        # iterate over fold combinations
        for fold in range(self.folds):
            train_idx = np.concatenate([split for i, split in enumerate(self.splits) if i != fold])
            valid_idx = self.splits[fold]
            
            #print(len(train_idx))
            #print(len(valid_idx))

            # return training and validation set as Subsets of the original dataset
            if self.subset_size:
                dataset_train = torch.utils.data.Subset(self.dataset, train_idx[:self.n_train])
                dataset_valid = torch.utils.data.Subset(self.dataset, valid_idx[:self.n_val])
            else:
                dataset_train = torch.utils.data.Subset(self.dataset, train_idx)
                dataset_valid = torch.utils.data.Subset(self.dataset, valid_idx)

            yield dataset_train, dataset_valid
    
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
        image, label = self.composed_trsfm((image, label))

        return image, label

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

def compute_mean_class_occurences(dataloader):

    cl_occurences = np.zeros(3)

    for _, mask_batch in dataloader:
        batch_np = mask_batch.view(mask_batch.size(0), -1).numpy()
        for i in range(3):
            cl_occurences[i] += np.mean(np.count_nonzero(batch_np == i, axis=1))
        
    return cl_occurences/len(dataloader)

    
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

    occurences = compute_mean_class_occurences(dataloader)
    print(occurences) #rounded [131383, 68638, 49979]
    #mean, std = compute_mean_and_std(dataloader, device="cuda:0") #set to "cpu" if not gpu is available
    #print(mean, std) #prints approx. mean=(0.7979, 0.6772, 0.7768), std=(0.1997, 0.3007, 0.2039)




   








