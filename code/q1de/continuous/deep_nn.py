import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.set_default_dtype(torch.float64)
class DeepNN(nn.Module):
    def __init__(self,depth,nbasis):
        super(DeepNN, self).__init__()
        forward_list = []
        self.nlayers = depth

        '''
        set input output dimension of the layers
        here use a simple rule where we double each layer
        and then go down to the desired latent state at the end
        '''
        dim = np.zeros(depth+2,dtype='int')
        dim[0] = 2
        for i in range(1,depth):
         dim[i] = dim[i-1]*2

        dim[-2] = nbasis
        dim[-1] = 1
        input_dim = dim[0:-1]
        output_dim = dim[1::]

        for i in range(0,self.nlayers):
          forward_list.append(nn.Linear(input_dim[i], output_dim[i]))

        forward_list.append(nn.Linear(input_dim[-1], output_dim[-1],bias=False))
        self.forward_list = nn.ModuleList(forward_list)
        self.activation = F.elu

    def createBasis(self,x):
      for i in range(0,self.nlayers):
        x = self.activation(self.forward_list[i](x))
      return x

    def forward(self,x):
      for i in range(0,self.nlayers):
        x = self.activation(self.forward_list[i](x))
      x = self.forward_list[-1](x)
      return x

