import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import utils.datasetscls as datasetscls
import utils.metrics as metrics
import os

def pretraining(
    model:torch.nn.Module, 
    dbgenerator:object, 
    batch_size: int=256,
    epochs: int=10,
    savepath: str = './save/models/', 
    device = 'cuda:0',
    savemodel: bool=True
    ):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.AE.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss = 0
        count = 0
        model.AE.train()  # Set model to training mode

        for _, filexy in enumerate(dbgenerator):
            if isinstance(filexy, tuple) and len(filexy) == 2:
                filex, _ = filexy
            else:
                filex = filexy


            dataset = datasetscls.customDataset(filex)
            sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

            for _, batch in enumerate(dataloader):
                if isinstance(batch, tuple) and len(batch) == 2:
                    x, _ = batch
                    x = x.to(device)
                else:
                    x = batch
                    x = x.to(device)


                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # forward
                    outputs = model.AE(x)
                    train_loss = criterion(outputs, x)
                    
                    # backward
                    train_loss.backward()
                    optimizer.step()


                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
                count+=1
            
        # compute the epoch training loss
        loss = loss / count
        print(f'epoch {epoch+1},loss = {loss:.8f}')

    if savemodel:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        torch.save(model.AE.state_dict(), os.path.join(savepath,'ae_weights.pth'))


def training(
    model:torch.nn.Module, 
    optimizer:torch.optim, 
    criterion:torch.nn, 
    y_pred_last:float,
    x:torch.tensor,
    y:torch.tensor=None,
    batch_size:int=256, 
    update_interval:int=30,
    device:str='cuda:0',
    update_freq:bool=False
    ):
    """
    """


    index_array = np.arange(x.shape[0])
    index = 0
    loss = 0
    count = 0
    for i in range(int(np.ceil(x.shape[0]/batch_size))):
        if i % update_interval == 0:
            with torch.no_grad():
                q = model(x)
                p = model.clustlayer.target_distribution(q)  # update the auxiliary target distribution p
                y_pred = q.argmax(1)

                if update_freq and i != 0 :
                    if y is not None:
                        acc = np.round(metrics.acc(y.clone().detach().cpu().numpy(), y_pred.clone().detach().cpu().numpy()), 5)
                        nmi = np.round(metrics.nmi(y.clone().detach().cpu().numpy().squeeze(), y_pred.clone().detach().cpu().numpy()), 5)
                        ari = np.round(metrics.ari(y.clone().detach().cpu().numpy().squeeze(), y_pred.clone().detach().cpu().numpy()), 5)
                        loss = np.round(loss/count, 5)
                        print('iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (i, acc, nmi, ari), ' ; loss=', loss)
                    else:
                        nmi = np.round(metrics.nmi(y_pred_last, y_pred.clone().detach().cpu().numpy()), 5)
                        ari = np.round(metrics.ari(y_pred_last, y_pred.clone().detach().cpu().numpy()), 5)
                        loss = np.round(loss/count, 5)
                        print('iter %d: nmi = %.5f, ari = %.5f' % (i, nmi, ari), ' ; loss=', loss)

                y_pred_last = y_pred.detach().clone().cpu().numpy()
            
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]

            trainx = x[idx]
            trainy = p[idx]

            trainx = trainx.to(device)
            trainy = trainy.to(device)

            outputs = model(trainx)
            index = index + 1 if (index + 1) * batch_size < x.shape[0] else 0

            train_loss = criterion(outputs.log(), trainy)

            train_loss.backward()
            optimizer.step()

            loss += train_loss.item()
            count +=1

    return loss/count


def testing(
    model: torch.nn.Module,
    dbgenerator: object,
    batch_size: int = 1024,
    device: str= 'cuda:0',
    return_truth: bool = True,
):
    """

    """

    preds = []
    gtruths = []
    for _, filexy in enumerate(dbgenerator):
        if isinstance(filexy, tuple) and len(filexy) == 2:
            filex, filey = filexy
            filex = filex.to(device)
            filey = filey.to(device)
        
        elif not isinstance(filexy, tuple):
            filex = filexy
            filex = filex.to(device)
            filey = None
        
        dataset = datasetscls.customDataset(filex, filey)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        for _, batch in enumerate(dataloader):
            x = batch
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                x, y = batch  
                if return_truth:
                    gtruths.append(y)      
            elif return_truth:
                raise ValueError(
                    "Dataset has no ground truth to return"
                )
            x = x.to(device)
            preds.append(
                model(x).detach().cpu()
            )  

    if return_truth:
        return torch.cat(preds).max(1)[1], torch.cat(gtruths).long()
    else:
        return torch.cat(preds).max(1)[1]