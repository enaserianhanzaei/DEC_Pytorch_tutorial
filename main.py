
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from utils import params
import os

from utils import datasetscls 
import DeepEncoderClustering
from train import training
from utils import metrics
from utils.utilityfn import getinputsize
from train import training, testing, pretraining

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datadir', default='/data/stl/fc1/')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--input_size', default=None)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--pretrain_epochs', default=10, type=int)
    parser.add_argument('--update_interval', default=30, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--save_dir', default='./save/')
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--save_intermodel', default=False, type=bool)

    args = parser.parse_args()
    print(args)

    #generator = datasetscls.customGenerator(args.datadir)
    generator = datasetscls.STLGenerator(args.datadir)
    try:
        assert len(generator)>=1
    except AssertionError:
        print('There should at least one input file.')
        raise

    if not args.input_size:
        args.input_size = getinputsize(generator)

    DEC = DeepEncoderClustering.DEC(inputsize=args.input_size, dims=params.dims, n_clusters=args.n_clusters)
    DEC.to(device)

    ae_weights = f'{args.save_dir}/models/stl/'
    if not os.path.exists(ae_weights):
        os.makedirs(ae_weights)

    if not os.path.exists(ae_weights+'ae_weights.pth'):
        pretraining(model=DEC, dbgenerator=generator, savepath=ae_weights, batch_size=args.batch_size, epochs=args.pretrain_epochs)
    else:
        DEC.AE.load_state_dict(torch.load(ae_weights+'ae_weights.pth'))

    DEC.train()  # Set model to training mode

    with torch.no_grad():
        print('Initializing cluster centers with k-means. number of clusters %s' % args.n_clusters)
            
        allfeatures = []
        for _, filexy in enumerate(generator):
            if isinstance(filexy, tuple) and len(filexy) == 2:
                filex, filey = filexy
                filex = filex.to(device)
                filey = filey.to(device)
            else:
                filex = filexy
                filex = filex.to(device)
            allfeatures.append(DEC.AE.encoder(filex).clone().detach().cpu())


        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
        y_pred_last = kmeans.fit_predict(torch.cat(allfeatures))
        seedfeatures, seedlabels=None, None
        
        clustcenters = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, requires_grad=True)
        clustcenters = clustcenters.to(device)

        DEC.state_dict()["clustlayer.clustcenters"].copy_(clustcenters)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.SGD(DEC.model.parameters(), lr=0.01, momentum=0.9)

    delta_label = None
    for epoch in range(args.epochs):
        loss = 0
        for _, filexy in enumerate(generator):
            if isinstance(filexy, tuple) and len(filexy) == 2:
                filex, filey = filexy
                filex = filex.to(device)
                filey = filey.to(device)
            
            elif not isinstance(filexy, tuple):
                filex = filexy
                filex = filex.to(device)
                filey = None
            
            train_loss = training(model=DEC, optimizer=optimizer, criterion=criterion, y_pred_last=y_pred_last, x=filex, y=filey, batch_size=args.batch_size, update_interval=args.update_interval, device = device)
            loss += train_loss

        if filey is not None:
            y_pred, acty = testing(model=DEC, dbgenerator=generator, device=device)
            acc = np.round(metrics.acc(acty.clone().detach().cpu().numpy(), y_pred.clone().detach().cpu().numpy()), 5)
            nmi = np.round(metrics.nmi(acty.clone().detach().cpu().numpy().squeeze(), y_pred.clone().detach().cpu().numpy()), 5)
            ari = np.round(metrics.ari(acty.clone().detach().cpu().numpy().squeeze(), y_pred.clone().detach().cpu().numpy()), 5)
            loss = np.round(loss/len(generator), 5)
            print('epoch %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (epoch, acc, nmi, ari), ' ; loss=', loss)
        else:
            y_pred = testing(model=DEC, dbgenerator=generator, device=device, return_truth=False)
            nmi = np.round(metrics.nmi(y_pred_last, y_pred.clone().detach().cpu().numpy()), 5)
            ari = np.round(metrics.ari(y_pred_last, y_pred.clone().detach().cpu().numpy()), 5)
            loss = np.round(loss/len(generator), 5)
            print('epoch %d: nmi = %.5f, ari = %.5f' % (epoch, nmi, ari), ' ; loss=', loss)

        delta_label = np.sum(y_pred_last!= y_pred.clone().detach().cpu().numpy()) / y_pred.shape[0]
        if args.tol is not None and delta_label < args.tol:
            print('delta_label ', delta_label, '< tol ', args.tol)
            print('Reached tolerance threshold. Stopping training.')
            break 

        y_pred_last = y_pred.detach().clone().cpu().numpy()

        if args.save_intermodel:
            torch.save(DEC.state_dict(), f'{args.save_dir}/models/dec_weights_%s_epoch%s.pth'%(args.n_clusters, epoch))

    if args.save_model:
        torch.save(DEC.state_dict(), f'{args.save_dir}/models/dec_weights_%s.pth'%(args.n_clusters))
        