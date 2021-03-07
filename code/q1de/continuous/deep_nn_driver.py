from pylab import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn_multistate import DeepNN 
import time
close("all")
torch.set_default_dtype(torch.float32)

#== Load data
U = np.load('../snapshots.npz')['snapshots'][:,:] 
N,nparams = np.shape(U)
nx = int(N/3)
U = np.reshape(U,(3,nx,nparams))
U = np.reshape(U,(3,nparams*nx))
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
depth=1
nbasis = 1
model = DeepNN(depth,nbasis,3)
np.random.seed(1)
try:
  model.load_state_dict(torch.load('tmp_model_depth' + str(depth) + '_nbasis_' + str(nbasis),map_location='cpu'))
  print('Succesfully loaded model!')
except:
  print('Failed to load model')

device = 'cpu'
model.to(device)


train_data = np.float32(np.append(input_features,U[:,:],axis=1))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100)

#Loss function
def my_criterion(y,yhat):
  loss_mse = torch.mean( (y - yhat[:,:,0])**2 )
  return loss_mse 
#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

 #Epochs
n_epochs = 20000
train_loss_hist = np.zeros(0)
t0 = time.time()
plot_freq = 100
for epoch in range(1, n_epochs+1):
    if (epoch == 200):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if (epoch == 1000):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if (epoch == 3000):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
      #figure(1)
      ax1 = plt.subplot(221)
      yhat_train = model.forward(torch.Tensor(test_input)).detach().numpy()
      ax1.plot(yhat_train[:,0].flatten(),color='blue',label='ML')
      ax1.plot(utest[0:1024,3].flatten(),color='red',label='Truth')
      ax1.set_xlabel(r'$x$')
      ax1.set_ylabel(r'$\rho(x)$')
      #savefig('rho_test_depth' + str(depth) + '.png')

      ax2 = plt.subplot(222)
      ax2.plot(yhat_train[:,1].flatten(),color='blue',label='ML')
      ax2.plot(utest[1024:1024*2,3].flatten(),color='red',label='Truth')
      ax2.set_xlabel(r'$x$')
      ax2.set_ylabel(r'$\rho U (x)$')
      #savefig('rhou_test_depth' + str(depth) + '.png')

      #figure(3)
      ax3 = plt.subplot(223)

      ax3.plot(yhat_train[:,2].flatten(),color='blue',label='ML')
      ax3.plot(utest[1024*2:1024*3,3].flatten(),color='red',label='Truth')
      ax3.set_xlabel(r'$x$')
      ax3.set_ylabel(r'$\rho E(x)$')
      savefig('sol_depth' + str(depth) + '_nbasis_' + str(nbasis) + '.png')

      ax4 = plt.subplot(224)
      ax4.loglog(train_loss_hist)
      ax4.set_xlabel(r'Epoch')
      ax4.set_ylabel(r'Loss')
      plt.tight_layout()
      savefig('training_depth' + str(depth) + '_nbasis_' + str(nbasis) + '.png')
      close("all")
      torch.save(model.state_dict(), 'tmp_model_depth' + str(depth) + '_nbasis_' + str(nbasis))

  
tf = time.time()


