import os
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class customDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y
        self.n_samples = self.X.shape[0]

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx]
        else:
            return self.X[idx]

    def __len__(self):
        return self.n_samples



class STLGenerator(object):
    def __init__(self, datapath):
        self.filenamesx, self.filenamesy = self.getXfiles(datapath)
 
    def __getitem__(self, idx):
        filenamex = self.filenamesx[idx]
        X = np.load(filenamex)
        X = torch.from_numpy(X.astype(np.float32))
        if self.filenamesy:
            filenamey = self.filenamesy[idx]
            y = np.load(filenamey)
            y = torch.from_numpy(y.astype(np.float32))
        else:
            y=None
        return X, y

    def __len__(self):
        return len(self.filenamesx)


    @staticmethod
    def getXfiles(datapath):
        allxpath = []
        allypath = []
        for root, _, files in os.walk((os.path.normpath(datapath)), topdown=False):
            dir = root.split('/')[-1]
            if dir != 'random':
                for name in files:
                    if name.endswith('x.npy'):
                        path = os.path.join(root, name)
                        allxpath.append(path)
                    elif name.endswith('y.npy'):
                        path = os.path.join(root, name)
                        allypath.append(path)
        return allxpath, allypath


class customGenerator(object):
    def __init__(self, datapath):
        self.filenamesx = self.getXfiles(datapath)
 
    def __getitem__(self, idx):
        filenamex = self.filenamesx[idx]
        X = np.load(filenamex)
        X = torch.from_numpy(X.astype(np.float32))
        return X

    def __len__(self):
        return len(self.filenamesx)


    @staticmethod
    def getXfiles(datapath):
        allxpath = []
        for root, _, files in os.walk((os.path.normpath(datapath)), topdown=False):
            for name in files:
                if name.endswith('.npy'):
                    path = os.path.join(root, name)
                    allxpath.append(path)

        return allxpath


class customImageFolder(datasets.ImageFolder):
    """Custom dataset that includes image file paths. 
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(customImageFolder, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

