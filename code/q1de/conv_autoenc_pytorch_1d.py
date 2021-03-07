from pylab import *
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

nvars = 1

#Define the Ceonvolutional Autoencoder
class ConvAutoencoder1d(nn.Module):
    def __init__(self,nchan,nx,num_latent_states):
        self.num_latent_states = num_latent_states
        self.nchan = nchan
        self.conv_depth = 5
        self.factor = int( 2**self.conv_depth )
        self.convFinalSize = int( nchan*self.factor*((nx)/self.factor  ))
        self.nx = nx
        super(ConvAutoencoder1d, self).__init__()
        #Encoder
        kernel_size = 25
        padding_size = int( (kernel_size - 1)/2 )

        kernel_sizef = kernel_size
        padding_sizef = int( (kernel_sizef - 1)/2 )

        kernel_size_tconv = int( kernel_size - 1 )
        padding_size_tconv = int( (kernel_size - 1)/2 - 1 )
        print(kernel_size_tconv)
        self.conv1 = nn.Conv2d(nchan, nchan*2, (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate') #  Nsamps x 8  x Nx/2 x Ny/2
        self.pool1 = nn.MaxPool2d( (2,1),padding=(0,0))
        self.conv2 = nn.Conv2d(nchan*2, nchan*4 , (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')   #  Nsamps x 16 x Nx/4 x Ny/4 
        self.pool2 = nn.MaxPool2d( (2,1),padding=(0,0))
        self.conv3 = nn.Conv2d(nchan*4 ,nchan*8 , (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')  #  Nsamps x 32 x Nx/8 x Ny/8 
        self.pool3 = nn.MaxPool2d( (2,1),padding=(0,0))
        self.conv4 = nn.Conv2d(nchan*8, nchan*16 , (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate') #   Nsamps x 64 x Nx/16 x Ny/16
        self.pool4 = nn.MaxPool2d( (2,1),padding=(0,0))
        self.conv5 = nn.Conv2d(nchan*16, nchan*32 , (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate') #   Nsamps x 64 x Nx/16 x Ny/16
        self.pool5 = nn.MaxPool2d( (2,1),padding=(0,0))

        #dfnn

        #Decoder
        self.t_conv0b = nn.Conv2d(nchan*32, nchan*16, (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')
        self.unpool0 = torch.nn.Upsample(scale_factor=(2,1),mode='nearest')


        self.t_conv1 = nn.ConvTranspose2d(nchan*16, nchan*8, (kernel_size_tconv,1), padding=(padding_size_tconv,0),stride=(2,1))
        self.t_conv1b = nn.Conv2d(nchan*16, nchan*8, (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')
        self.unpool1 = torch.nn.Upsample(scale_factor=(2,1),mode='nearest')

        self.t_conv2 = nn.ConvTranspose2d(nchan*8, nchan*4, (kernel_size_tconv,1), padding=(padding_size_tconv,0),stride=(2,1))
        self.t_conv2b = nn.Conv2d(nchan*8, nchan*4, (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')
        self.unpool2 = torch.nn.Upsample(scale_factor=(2,1),mode='nearest')

        self.t_conv3 = nn.ConvTranspose2d(nchan*4, nchan*2, (kernel_size_tconv,1), padding=(padding_size_tconv,0),stride=(2,1))
        self.t_conv3b = nn.Conv2d(nchan*4, nchan*2, (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')
        self.unpool3 = torch.nn.Upsample(scale_factor=(2,1),mode='nearest')

        self.t_conv4 = nn.ConvTranspose2d(nchan*2, nchan, (kernel_size_tconv,1), padding=(padding_size_tconv,0),stride=(2,1))
        self.t_conv4b = nn.Conv2d(nchan*2, nchan*1, (kernel_size,1), padding=(padding_size,0),stride=(1,1),padding_mode='replicate')
        self.unpool4 = torch.nn.Upsample(scale_factor=(2,1),mode='nearest')


        self.t_conv5 = nn.ConvTranspose2d(nchan, nchan, (kernel_size_tconv-1,1), padding=(padding_size_tconv,0),stride=(1,1))
        self.t_conv5b = nn.Conv2d(nchan, nchan, (kernel_sizef,1), padding=(padding_sizef,0),stride=(1,1),padding_mode='replicate')

        #self.t_conv1.weight.data.fill_(2*wf)
        self.conv1.bias.data.fill_(0.)
        self.conv2.bias.data.fill_(0.)
        self.conv3.bias.data.fill_(0.)
        self.conv4.bias.data.fill_(0.)

        self.t_conv1.bias.data.fill_(0.)
        self.t_conv2.bias.data.fill_(0.)
        self.t_conv3.bias.data.fill_(0.)
        self.t_conv4.bias.data.fill_(0.)
        self.t_conv5.bias.data.fill_(0.)
        self.t_conv1b.bias.data.fill_(0.)
        self.t_conv2b.bias.data.fill_(0.)
        self.t_conv3b.bias.data.fill_(0.)
        self.t_conv4b.bias.data.fill_(0.)
        self.t_conv5b.bias.data.fill_(0.)

        self.l1 = nn.Linear(self.convFinalSize,self.num_latent_states)
        self.l2 = nn.Linear(self.num_latent_states,self.convFinalSize)
        self.l1.bias.data.fill_(0.)
        self.l2.bias.data.fill_(0.)


    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = x.view(x.size(0),1,self.convFinalSize)
        x = F.relu(self.l1(x))
        return x

 
    def decoderb(self, x):
        x = F.relu(self.l2(x))
        x = x.view(int(x.size(0)),self.nchan*self.factor,int(self.nx/self.factor),1)
        x = self.unpool0(x)
        x = F.relu(self.t_conv0b(x))

        x = self.unpool1(x)
        x = F.relu(self.t_conv1b(x))
        x = self.unpool2(x)
        x = F.relu(self.t_conv2b(x))
        x = self.unpool3(x)
        x = F.relu(self.t_conv3b(x))
        x = self.unpool4(x)
        x = F.relu(self.t_conv4b(x))
        x = self.t_conv5b(x)
        return x


    def decoder(self, x):
        x = F.relu(self.l2(x))
        x = x.view(int(x.size(0)),self.nchan*self.factor,int(self.nx/self.factor),1)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = self.t_conv5(x)
        return x


    def forward(self, x):
        z = self.encoder(x)
        x = self.decoderb(z)
        return x

