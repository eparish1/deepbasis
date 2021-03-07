from pylab import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn import DeepNN 
import time
close("all")
torch.set_default_dtype(torch.float32)

#
varTrain = "rhoE"
if varTrain == "rho":
  indx = 0
if varTrain == "rhoU":
  indx = 1
if varTrain == "rhoE":
  indx = 2

#== Load data
U = np.load('../snapshots.npz')['snapshots'][indx*1024:(indx+1)*1024,:] 
N,nparams = np.shape(U)
nx = int(N)
U = np.reshape(U,(1,nx,nparams))
U = np.reshape(U,(1,nparams*nx))
U = np.rollaxis(U,1)

#create input parameters
x = np.linspace(0,1,nx)
params = np.load('../snapshots.npz')['params']
x,params = meshgrid(x,params,indexing='ij')
x = x.flatten()
params = params.flatten()
input_features = np.append(x[:,None],params[:,None],axis=1)


utest = np.load('../snapshots_fine.npz')['snapshots']
params_test = np.load('../snapshots_fine.npz')['params']
x_test = linspace(0,1,nx)

x_test,params_test = meshgrid(x_test,params_test,indexing='ij')
x_test = x_test.flatten()
params_test = params_test.flatten()
input_features_test = np.append(x_test[:,None],params_test[:,None],axis=1)

## create a test instance for online plotting
test_input = np.zeros((nx,2))
test_input[:,0] = np.linspace(0,1,nx)
test_input[:,1] = params_test[3]


#===
depth=4
nbasis=4
model = DeepNN(depth,nbasis)
np.random.seed(1)
try:
  model.load_state_dict(torch.load(varTrain + '_model_depth' + str(depth),map_location='cpu'))
  print('Succesfully loaded model!')
except:
  print('Failed to load model')

device = 'cpu'
model.to(device)


train_data = np.float32(np.append(input_features,U,axis=1))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100)

#Loss function
def my_criterion(y,yhat):
  loss_mse = torch.mean( (y - yhat)**2 )
  return loss_mse 
#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

 #Epochs
n_epochs = 20000
train_loss_hist = np.zeros(0)
t0 = time.time()
plot_freq = 100
for epoch in range(1, n_epochs+1):
    if (epoch == 400):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if (epoch == 2000):
      optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    if (epoch == 3000):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # monitor training loss
    train_loss = 0.0
    #Training
    for data in train_loader:
        data_d = data.to(device,dtype=torch.float32)
        inputs = data_d[:,0:2]
        y = data_d[:,2::]
        optimizer.zero_grad()
        yhat = model(inputs)
        loss = my_criterion(y,yhat)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*inputs.size(0)

  
    train_loss = train_loss#/len(train_loader)
    train_loss_hist = np.append(train_loss_hist,train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    print('Time: {:.6f}'.format(time.time() - t0))

    if (epoch%plot_freq == 0):        
      figure(1)
      yhat_train = model.forward(torch.Tensor(test_input)).detach().numpy()
      plot(yhat_train[:,0].flatten(),color='blue',label='ML')
      plot(utest[indx*nx:(indx+1)*nx,3].flatten(),color='red',label='Truth')
      xlabel(r'$x$')
      ylabel(varTrain)
      savefig(varTrain + '_test_depth' + str(depth) + '.png')
      close("all")
      torch.save(model.state_dict(), varTrain + '_model_depth' + str(depth))

  
