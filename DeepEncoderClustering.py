
import torch
import torch.nn as nn
from typing import Optional, List
import math

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


class autoencoder(nn.Module):
    def __init__(
        self, 
        inputsize: int,
        dims: List[int]):
        """
        
        """
        
        super(autoencoder, self).__init__()

        self.inputsize = inputsize

        encmodules = []
        encmodules.append(nn.Linear(inputsize, dims[0]))
        for index in range(len(dims)-1):
            encmodules.append(nn.ReLU(True))
            encmodules.append(nn.Linear(dims[index], dims[index+1]))
        self.encoder = nn.Sequential(*encmodules)

        decmodules = []
        for index in range(len(dims) - 1, 0, -1):
            decmodules.append(nn.Linear(dims[index], dims[index-1]))
            decmodules.append(nn.ReLU(True))
        decmodules.append(nn.Linear(dims[0], inputsize))
        self.decoder = nn.Sequential(*decmodules)

        self.init_weights()

    def forward(
        self, 
        x
        ):
        """
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_encoder(
        self
        ):
        """
        """
        return self.encoder

    def init_weights(
        self
        ):
        """
        """
        #glorot_uniform . Draws samples from a uniform distribution within [-limit, limit] , where limit = sqrt(6 / (fan_in + fan_out)) 
        def func(m):
            if isinstance(m, nn.Linear):
                torch.manual_seed(4)
                limit = math.sqrt(6/(m.in_features + m.out_features))
                torch.nn.init.uniform_(m.weight, -limit, limit)
                m.bias.data.fill_(0.00)

        self.encoder.apply(func)
        self.decoder.apply(func)



class clustering(nn.Module):

    def __init__(
        self, 
        n_clusters:int,
        input_shape:int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None
        ) -> None:
        """
        """

        super(clustering, self).__init__()

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.input_shape = input_shape

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.n_clusters, self.input_shape, dtype=torch.float32)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.clustcenters = nn.Parameter(initial_cluster_centers)



    def forward(self, inputs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, axis=1) - self.clustcenters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, axis=1), 0, 1)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T


class DEC(nn.Module):
    def __init__(
        self, 
        dims: List[int],
        inputsize: int, 
        n_clusters: int):
        """
        """
        super(DEC, self).__init__()
        self.AE = autoencoder(inputsize, dims)
        self.clustlayer = clustering(n_clusters, dims[-1])

        self.model = nn.Sequential(
            self.AE.encoder,
            self.clustlayer)    
        
    def forward(
        self, 
        inputs):
        """
        """

        X = self.model(inputs)
        return X
    