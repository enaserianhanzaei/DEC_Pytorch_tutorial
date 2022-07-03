import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def meanstd_torchVGG16(args, verbose=False):
    tra = [transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]
    dataset = datasets.ImageFolder(args.datadir, transform=transforms.Compose(tra))
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch,
                                            num_workers=args.workers,
                                            pin_memory=True)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for _, (input_tensor, _) in enumerate(dataloader):
        input_tensor_ = input_tensor.detach().clone().to(device)
        input_tensor_ = input_tensor_.view(args.batch, input_tensor_.size()[1], -1)
        mean += input_tensor_.mean(2).sum(0)
        std += input_tensor_.std(2).sum(0)

        nb_samples += args.batch

    mean /= nb_samples
    std /= nb_samples

    if verbose:
        print(f'mean and std for the this dataset is {mean}, and {std}')
    return torch.round(mean,decimals=3), torch.round(std,decimals=3)


def getinputsize(generator):
    for _, filexy in enumerate(generator):
        filex = filexy
        if (isinstance(filexy, tuple) or isinstance(filexy, list)) and len(filexy) == 2:
            filex, _ = filexy

        try:
            assert len(filex.shape) ==2
        except AssertionError:
            print("")
            raise

        inputsize=filex.shape[1]

        return inputsize

def readseeds(datapath):
    features = np.load(datapath + 'featuresx.npy')
    features = torch.from_numpy(features.astype(np.float32))

    pathfile = open(datapath+'/featuresy.txt', "r")
    pathlist = pathfile.readlines()
    pathlist = [path[:-1] for path in pathlist]
    pathfile.close()
    dictlabel = dict(zip(list(set(pathlist)), range(len(set(pathlist)))))
    labels = np.array([dictlabel[label] for label in pathlist])

    return features, labels




