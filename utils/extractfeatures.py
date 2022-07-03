
import torchvision.transforms as transforms
import argparse
import os
import numpy as np
import torch
from utils.transfermodels import PytorchVGG, VGG16normalization
from utils.datasetscls import customImageFolder
from utils.utilityfn import meanstd_torchVGG16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Feature extraction using Pytorch pretrained models')

parser.add_argument('--datadir', metavar='DDIR', type=str,
                    default='/home/elahe/NortfaceProject/codes/DEC-keras/results/clusters/Telegram_final/2021_AprilMayJun/VGG16/block5pool/normalisedfiltered/seeds/Stephane/',
                    help='path to dataset') 
parser.add_argument('--premodel', '-a', type=str, metavar='PMODEL',
                    choices=['alexnet', 'vgg16'], default='vgg16',
                    help='Pre-trained model (default: vgg16)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--conv', default=5, help='')
parser.add_argument('--fc', default=None, help='')
parser.add_argument('--ave_pool', default=True, type=bool,
                    help='')
parser.add_argument('--datasetnorm', default=True, type=bool,
                    help='whether normalize the input data mean and std, or based on VGG16 statistics')
parser.add_argument('--split', default=False, type=bool,
                    help='whether split the data into smaller files')
parser.add_argument('--splitsize', default=5120, type=int,
                    help='In case of splitting, the size of the split')
parser.add_argument('--savedir', metavar='SDIR', type=str,
                    default='./save',
                    help='path to saving directory') 

def main():
    global args
    args = parser.parse_args()

    if args.datasetnorm:
        #mean, std = meanstd_torchVGG16(args)
        mean = [0.553, 0.524, 0.506]#[0.549, 0.525, 0.515]
        std = [0.248, 0.254, 0.243]#[0.249, 0.243, 0.243]
        tra = VGG16normalization(mean, std)
    else:
        tra = VGG16normalization()

    # load the data
    dataset = customImageFolder(args.datadir, transform=transforms.Compose(tra))
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch,
                                            num_workers=args.workers,
                                            pin_memory=True)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    model = PytorchVGG(conv=args.conv, fc=args.fc, av_pool=args.ave_pool)
    model.eval()
    model.to(device)


    if not args.split:
        image_paths = []
        # discard the label information in the dataloader
        for i, (input_tensor, _, path) in enumerate(dataloader):
            image_paths.extend(path)
            input_var = input_tensor.clone()
            input_var = input_var.to(device)

            aux = model(input_var)
            aux = aux.data.cpu().numpy() 

            if i == 0:
                features = np.zeros((len(dataset), aux.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux
            else:
                # special treatment for final batch
                features[i * args.batch:] = aux

            if (i % 100) == 0 and i != 0:
                print(f'{i} batch have been computed')

        np.save(args.savedir + '/featuresx', features)

        with open(args.savedir + '/paths.txt', 'w') as f:
            for item in image_paths:
                f.write("%s\n" % item)
    else:
        image_paths = []
        j = 0
        count = 0
        # discard the label information in the dataloader
        for i, (input_tensor, _, path) in enumerate(dataloader):
            saved = False
            image_paths.extend(path)
            input_var = input_tensor.clone()
            input_var = input_var.to(device)

            input_var = model(input_var)
            input_var = input_var.data.cpu().numpy() 
            input_var = input_var.astype('float32')
            if j == 0:
                if i + np.ceil(args.splitsize/args.batch) < len(dataloader) - 1:
                    features = np.zeros((args.splitsize, input_var.shape[1]))
                else:
                    features = np.zeros((len(dataset) - (i * args.batch), input_var.shape[1]))

            
            if i < len(dataloader) - 1:
                features[j * args.batch: (j + 1) * args.batch] = input_var
            else:
                # special treatment for final batch
                features[j * args.batch:] = input_var

            count+=args.batch
            j+=1
            
            if count % (args.splitsize) == 0:
                print(f'{count} samples are computed')
                np.save(args.savedir + '/featuresx_%s' % (int(np.ceil(count / (args.splitsize)))), features)
                with open(args.savedir + '/paths_%s.txt' % (int(np.ceil(count / (args.splitsize)))), 'w') as f:
                    for item in image_paths:
                        f.write("%s\n" % item)
                j = 0
                image_paths = []
                saved = True

        # saving the final split
        if not saved:
            np.save(args.savedir + '/featuresx_%s' % (int(np.ceil(count / (args.splitsize)))), features)
            with open(args.savedir + '/paths_%s.txt' % (int(np.ceil(count / (args.splitsize)))), 'w') as f:
                for item in image_paths:
                    f.write("%s\n" % item)

    return features, image_paths


if __name__ == '__main__':
    main()