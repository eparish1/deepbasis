from pylab import *
#from mpi4py import MPI
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
from nlsvd import NLSVD 
close("all")
torch.set_default_dtype(torch.float64)

#torch.manual_seed(1)
#def build_and_train_model(var_start,train_no):
num_latent_states = 10 
U = np.load('snapshots_fine.npz')['snapshots'][:,:][0:1024,:,None] #nx x nt x nparams
Phi,d,d = np.linalg.svd(U[:,:,0])
Phi = Phi[:,0:num_latent_states]
U_project = np.rollaxis(np.dot(Phi,np.dot(Phi.transpose(),U[:,:,0])),1)
nx,nt,nparams = np.shape(U)
U = np.rollaxis(np.rollaxis(U,2,0),2,1) #nparams x nt x nx
U = np.reshape(U,(nparams*nt,nx))[:,None,:,None]

UT = np.load('snapshots_vfine.npz')['snapshots'][:,:][0:1024,:,None] #nx x nt x nparams
UT_project = np.rollaxis(np.dot(Phi,np.dot(Phi.transpose(),UT[:,:,0])),1)

nxT,ntT,nparamsT = np.shape(UT)
UT = np.rollaxis(np.rollaxis(UT,2,0),2,1) #nparams x nt x nx
UT = np.reshape(UT,(nparamsT*ntT,nxT))[:,None,:,None]



model = NLSVD(1,nx,num_latent_states,U[:])
try:
  model.load_state_dict(torch.load('tmp_model_svd',map_location='cpu'))
  print('Succesfully loaded model!')
except:
  print('Failed to load model')
device = 'cpu'
model.to(device)


U_ref = 0.#np.mean(U)
U_scale = 1.#np.amax(U) - np.amin(U)
utrain = (U - U_ref)/U_scale
train_loader = torch.utils.data.DataLoader(utrain, batch_size=5)

nxG = 128
xG = linspace(0,100,nxG)
dx = xG[1]
#Loss function
def my_criterion(outputs,inputs): 
  #outputs_dx1 = 0.5/dx*(outputs[:,:,2::,:] - outputs[:,:,0:-2] )
  #inputs_dx1 = 0.5/dx*(inputs[:,:,2::,:] - inputs[:,:,0:-2] )
  #loss_dx4 = torch.mean( (outputs[:,:,4::,:] - 4.*outputs[:,:,3:-1,:] + 6.*outputs[:,:,2:-2,:] - 4.*outputs[:,:,1:-3,:] + outputs[:,:,0:-4] )**2 ) 
  loss_mse = torch.mean( (outputs - inputs)**2 ) 
  #loss_mse_dx1 = torch.mean( (outputs_dx1 - inputs_dx1)**2 )  

  #print(loss_dx4,loss_mse)
  return loss_mse# + 10.*loss_mse_dx1# + loss_dx4
#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

 #Epochs
n_epochs = 14000
#inputs = torch.as_tensor(utrain[:],dtype=torch.float)
plot_freq = 1
train_loss_hist = np.zeros(0)
t0 = time.time()
for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training

    for data in train_loader:
        inputs = data.to(device,dtype=torch.float64)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = my_criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*inputs.size(0)
        
    yhat_train = model.forward(torch.Tensor(utrain)).detach().numpy()
    yhat_test = model.forward(torch.Tensor(UT)).detach().numpy()

    #yhat = model.forward(torch.Tensor(Utest)).detach().numpy()
    plot(yhat_train[-1].flatten(),color='blue')
    plot(utrain[-1,:,0:-1].flatten(),'--',color='blue')
    plot(U_project[-1].flatten(),'-.',color='green')
    plot(yhat_test[1].flatten(),color='red')
    plot(UT[1,:,0:-1].flatten(),'--',color='red')
    plot(UT_project[1].flatten(),'-.',color='green')
    print('Test error = ' + str(np.linalg.norm(yhat_test - UT)))
    print('PCA Test error = ' + str(np.linalg.norm(UT_project[:,None,:,None] - UT)))

    #plot(yhat[int(nt_test/2)].flatten(),color='red')
    #plot(Utest[int(nt_test/2),:,0:-1].flatten(),'--',color='red')
    pause(0.001)
    clf()
    
    train_loss = train_loss/len(train_loader)
    train_loss_hist = np.append(train_loss_hist,train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    print('Time: {:.6f}'.format(time.time() - t0))

#return model,train_loss




'''

  
n_training_runs = 1
train_loss_summary = np.zeros((5,n_training_runs))
models = [None]*1
for i in range(0,1):
  models[i] = [None]*n_training_runs

for i in range(0,1):
  for train_no in range(0,n_training_runs):
    model,train_loss = build_and_train_model(i,train_no)
    models[i][train_no] = model
    train_loss_summary[i,train_no] = train_loss

'''
