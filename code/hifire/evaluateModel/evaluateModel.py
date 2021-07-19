from pylab import *
import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn_multistate import DeepNN 
import time
close("all")
torch.set_default_dtype(torch.float32)

def evaluateModel(modelName,Normalize=False,BoxInit=False):
    #== Load data
    training_data = np.load('../training_data.npz')
    testing_data = np.load('../training_data.npz')

    U = testing_data['U'] 
    UTrain = training_data['U'] 

    if Normalize:
      x_shift = np.amin(training_data['X'][:,0])
      x_scale = np.amax(training_data['X'][:,0]) - np.amin(training_data['X'][:,0])
      y_shift = np.amin(training_data['X'][:,1])
      y_scale = np.amax(training_data['X'][:,1]) - np.amin(training_data['X'][:,1])
      mu1_shift = np.amin(training_data['params'][:,0])
      mu1_scale = np.amax(training_data['params'][:,0]) - np.amin(training_data['params'][:,0])
      mu2_shift = np.amin(training_data['params'][:,1])
      mu2_scale = np.amax(training_data['params'][:,1]) - np.amin(training_data['params'][:,1])
  
      x = (testing_data['X'][:,0] - np.amin(training_data['X'][:,0]))/( np.amax(training_data['X'][:,0] - np.amin(training_data['X'][:,0])) )
      y = (testing_data['X'][:,1] - np.amin(training_data['X'][:,1]))/( np.amax(training_data['X'][:,1] - np.amin(training_data['X'][:,1])) )
      mu1 = (testing_data['params'][:,0] - np.amin(training_data['params'][:,0]))/( np.amax(training_data['params'][:,0]) - np.amin(training_data['params'][:,0]))
      mu2 = (testing_data['params'][:,1] - np.amin(training_data['params'][:,1]))/( np.amax(training_data['params'][:,1]) - np.amin(training_data['params'][:,1]))

      for i in range(0,6):
        U[:,i::6] = (U[:,i::6] - np.amin(UTrain[:,i::6]) ) / ( np.amax(UTrain[:,i::6]) - np.amin(UTrain[:,i::6]))

    else:
      x = testing_data['X'][:,0]
      y = testing_data['X'][:,1]
      mu1 = testing_data['params'][:,0]
      mu2 = testing_data['params'][:,1]

    nparams,N =  np.shape(U)
    nc = 6
    U2 = np.zeros((6,nparams,int(N/6)))
    for i in range(0,6):
      U2[i,:,:] = U[:,i::6]
    #U = np.reshape(U2,(6,nparams*int(N/6))) #will run through all x, then params
    #U = np.rollaxis(U,1)
    nx = np.size(x)
    nparams = np.size(mu1)
    xv = np.tile(x,nparams)# use tile here to run in order
    yv = np.tile(y,nparams) 
    mu1v = np.repeat(mu1,nx) #run through all nx/ny and nt before incrementing mu
    mu2v = np.repeat(mu2,nx) #run through all nx/ny and nt before incrementing mu

    input_features = np.append(xv[:,None],yv[:,None],axis=1)
    input_features = np.append(input_features,mu1v[:,None],axis=1)
    input_features = np.append(input_features,mu2v[:,None],axis=1)
    input_features_size = np.size(input_features[:,0])
    #===
    model = DeepNN(depth,nbasis,nc,BoxInit)
    model.load_state_dict(torch.load('../tmpdepth_' + str(depth) + '_nbasis_' + str(nbasis) + '_no_0',map_location='cpu'))

    device = 'cpu'
    model.to(device)
    
    r1 = model.forward(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()[:,:,0]
    r1 = np.rollaxis(r1,1)
    r1 = np.reshape(r1,(6,nparams,int(N/6)))
    return r1,U2
depth_a = np.array([7])
nbasis_a = np.array([8])
nmodels = 1
NormalizeData = True
BoxInit = False

X  = np.load('../training_data.npz')['X']

for i in range(0,nmodels):
 for nbasis in nbasis_a:
  for depth in depth_a:
    print("=======================================================")
    print("=======================================================")
    print("Training the " + str(i) + " model with basis dimension " + str(nbasis))
    print("=======================================================")
    print("=======================================================")

    modelName = 'depth_' + str(depth) + '_nbasis_' + str(nbasis) + '_no_' + str(i) 
    yhat,y=evaluateModel(modelName,NormalizeData,BoxInit)
