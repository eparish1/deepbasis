import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
class ShallowAutoencoder(nn.Module):
    def __init__(self,nchan,nx,num_latent_states,snapshots,depth=1):
        self.num_latent_states = num_latent_states
        self.nchan = nchan
        self.nx = nx
        super(ShallowAutoencoder, self).__init__()
        #Encoder
        kernel_size = 31 
        padding_size = int( (kernel_size - 1)/2 )

        forward_list = []
        backward_list = []

        self.nlayers = depth
        dim = np.zeros(depth+1,dtype='int')
        for i in range(0,depth):
          scale = 2**i
          dim[i] = nx/scale
        dim[-1] = num_latent_states
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        #if (depth == 1):
        #  input_dim = np.array([nx])
        #  output_dim = np.array([num_latent_states])
#
#        if (depth == 2):
#          input_dim = np.array([nx,int(nx/2)])
#          output_dim = np.array([int(nx/2),num_latent_states])

        for i in range(0,self.nlayers):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))
          backward_list.append(nn.Linear(output_dim[-i-1], input_dim[-i-1]))

        self.forward_list = nn.ModuleList(forward_list)
        self.backward_list = nn.ModuleList(backward_list)

        self.forward_list = nn.ModuleList(forward_list)
        self.backward_list = nn.ModuleList(backward_list)
        self.final = nn.Conv2d(nchan, nchan, (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')

        self.activation = F.elu
        self.snapshots = torch.tensor( snapshots[:,0,:,0] )

    def encoder(self,x):
      for i in range(0,self.nlayers):
        x = self.activation(self.forward_list[i](x))
      return x

    def decoder(self,x):
      for i in range(0,self.nlayers):
        x = self.activation(self.backward_list[i](x))
      x = x[:,None,:,None]
      x = self.final(x)
      return x[:,0,:,0]

    def forward(self, x):
        x = x[:,0,:,0]
        x = self.decoder(self.encoder(x))
        return x[:,None,:,None]


