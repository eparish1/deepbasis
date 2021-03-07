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
from conv_autoenc_pytorch_1d import ConvAutoencoder1d
torch.set_default_dtype(torch.float64)
class NLSVD(nn.Module):
    def __init__(self,nchan,nx,num_latent_states,snapshots):
        self.num_latent_states = num_latent_states
        self.nchan = nchan
        self.conv_depth = 5
        self.factor = int( 2**self.conv_depth )
        self.convFinalSize = int( nchan*self.factor*((nx)/self.factor  ))
        self.nx = nx
        super(NLSVD, self).__init__()
        #Encoder
        kernel_size = 31 
        padding_size = int( (kernel_size - 1)/2 )

        forward_list = []
        backward_list = []

        self.nlayers = 4
        #nchan_a = np.ones(self.nlayers,dtype='int')
        nchan_a = np.array([1,2,4,2,1])
        for i in range(0,self.nlayers):
          forward_list.append(nn.Conv2d(nchan_a[i], nchan_a[i+1], (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate'))
          backward_list.append(nn.Conv2d(nchan_a[-i-1], nchan_a[-i-2], (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate'))

        self.forward_list = nn.ModuleList(forward_list)
        self.backward_list = nn.ModuleList(backward_list)
#        for i in range(0,self.nlayers):
#          forward_list[i].weight.data.fill_(1./kernel_size)
#          backward_list[i].weight.data.fill_(1./kernel_size)

        self.forward_list = nn.ModuleList(forward_list)
        self.backward_list = nn.ModuleList(backward_list)
        self.final = nn.Conv2d(nchan, nchan, (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')

        self.activation = F.elu
        self.snapshots = torch.tensor( snapshots )
    def transform(self,x):
      for i in range(0,self.nlayers):
        x = self.activation(self.forward_list[i](x))
      return x

    def encoder(self,x):
      if (np.size(np.shape(x))==1):
        multiInput = False
        x = self.transform(torch.tensor(x[None,None,:,None]))
      else:
        multiInput = True
        x = self.transform(torch.tensor(x[:,None,:,None]))
      x = x[:,0,:,0]
      x = x.transpose(0,1)
      xhat = torch.mm(self.basis[:,0:self.num_latent_states].transpose(0,1),x)
      if (multiInput==False):
        return xhat[:,:].detach().numpy().flatten()
      else:
        return np.rollaxis(xhat[:,:].detach().numpy(),1) 


    def decoder(self,xhat):
      if (np.size(np.shape(xhat))==1):
        multiInput = False
        x = torch.mm(self.basis[:,0:self.num_latent_states],torch.tensor(xhat[:,None]))
      else:
        multiInput = True
        x = torch.mm(self.basis[:,0:self.num_latent_states],torch.tensor(np.rollaxis(xhat[:,:],1)))
      x = x.transpose(0,1)[:,None,:,None]
      x = self.inverse_transform(x)
      if multiInput==False:
        return x.detach().numpy().flatten()
      else:
        return x.detach().numpy()[:,0,:,0]
   
    def inverse_transform(self,x):
      for i in range(0,self.nlayers):
        x = self.activation(self.backward_list[i](x))
      x = self.final(x)
      return x

    def forward(self, x):
        snapshots = self.transform(self.snapshots)
        snapshots = snapshots[:,0,:,0]
        snapshots = snapshots.transpose(0,1)

        x = self.transform(x)
        x = x[:,0,:,0]
        x = x.transpose(0,1)

        u,s,v = torch.linalg.svd(snapshots,full_matrices=True)
        self.basis = u[:,0:self.num_latent_states]
        xhat = torch.mm(u[:,0:self.num_latent_states].transpose(0,1),x)
        x = torch.mm(u[:,0:self.num_latent_states],xhat)
        x = x.transpose(0,1)[:,None,:,None]
        x = self.inverse_transform(x)
        return x

    def setBasis(self,snapshots):
        snapshots = self.transform(self.snapshots)
        snapshots = snapshots[:,0,:,0]
        snapshots = snapshots.transpose(0,1)
        u,s,v = torch.linalg.svd(snapshots,full_matrices=True)
        self.basis = u[:,0:self.num_latent_states]

