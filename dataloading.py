import torch
import numpy as np

# for testing
import os
from CRC_Dataset import CRC_Dataset
from utils import MirrorPad
from torchvision.transforms import ToTensor


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


if __name__ == "__main__":

    dataset = CRC_Dataset(
        root_dir = os.path.join(os.getcwd(), 'data\\train'),
        transforms = [MirrorPad(((6,), (6,), (0,))), ToTensor()]
    )

    cv_sampler = CV_Splits(5, shuffle=True, dataset=dataset)
    print(cv_sampler.folds)
    
    cv_sampler(dataset)
    print(cv_sampler.dataset)

    d_t, d_v = next(iter(cv_sampler))
    print(d_t[0], d_v[0])
