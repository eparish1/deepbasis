from pylab import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn import DeepNN 
import time
close("all")
torch.set_default_dtype(torch.float32)

#== Load data
U = np.rollaxis( np.load('../snapshots.npz')['snapshots'][:,0:256:4] ,1) 
nt,nx= np.shape(U)
U = np.reshape(U,(1,nt,nx))
U = np.reshape(U,(1,nt*nx))
U = np.rollaxis(U,1)

#create input parameters
x = np.load('../snapshots.npz')['x']
t = np.load('../snapshots.npz')['t'][0:256:4]
x,t = meshgrid(x,t)
x = x.flatten()
t = t.flatten()
input_features = np.append(x[:,None],t[:,None],axis=1)


xtest = np.linspace(0,1,nx)
t = np.linspace(0,1,nt)
xtest = np.load('../snapshots.npz')['x']
ttest = np.load('../snapshots.npz')['t'][400:401]

xtest,ttest = np.meshgrid(xtest,ttest)
xtest = xtest.flatten()
ttest = ttest.flatten()
input_features_test = torch.tensor(np.append(xtest[:,None],ttest[:,None],axis=1),dtype=torch.float32)
Utest = np.load('../snapshots.npz')['snapshots'][:,400]


#===
depth=4
nbasis = 4
model = DeepNN(depth,nbasis)
np.random.seed(1)
try:
  model.load_state_dict(torch.load('tmp_model_depth' + str(depth),map_location='cpu'))
  print('Succesfully loaded model!')
except:
  print('Failed to load model')

device = 'cpu'
model.to(device)


train_data = np.float32(np.append(input_features,U,axis=1))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=500)

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
plot_freq = 50
for epoch in range(1, n_epochs+1):
    if (epoch == 200):
      optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    if (epoch == 2000):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
      yhat_train = model.forward(input_features_test).detach().numpy()
      plot(yhat_train.flatten(),color='blue',label='ML')
      plot(Utest.flatten(),color='red',label='Truth')
      xlabel(r'$x$')
      ylabel(r'$u(x)$')
      savefig('u_test_depth' + str(depth) + '.png')
      figure(2)
      loglog(train_loss_hist)
      xlabel(r'Epoch')
      ylabel(r'Loss')
      savefig('train_loss_depth' + str(depth) + '.png')
      close("all")
      torch.save(model.state_dict(), 'tmp_model_depth' + str(depth))

  
tf = time.time()

## Final plots
yhat = model(torch.tensor(input_features))
yhat_test = model(torch.tensor(input_features_test))
