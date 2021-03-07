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
U = np.load('../snapshots.npz')['snapshots'][:,:] 
nx,nt= np.shape(U)
U = np.reshape(U,(1,nx,nt))
U = np.reshape(U,(1,nt*nx))
U = np.rollaxis(U,1)

#create input parameters
x = np.linspace(0,1,nx)
t = np.linspace(0,1,nt)
x,t = meshgrid(x,t,indexing='ij')
x = x.flatten()
params = params.flatten()
input_features = np.append(x[:,None],t[:,None],axis=1)



#===
depth=4
model = DeepNN(depth)
np.random.seed(1)
try:
  model.load_state_dict(torch.load('tmp_model_depth' + str(depth),map_location='cpu'))
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

 #Epochs
n_epochs = 20000
train_loss_hist = np.zeros(0)
t0 = time.time()
plot_freq = 100
for epoch in range(1, n_epochs+1):
    if (epoch == 200):
      optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    if (epoch == 1000):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    if (epoch == 3000):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

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
      plot(utest[0:1024,3].flatten(),color='red',label='Truth')
      xlabel(r'$x$')
      ylabel(r'$\rho(x)$')
      savefig('rho_test_depth' + str(depth) + '.png')

      figure(2)
      plot(yhat_train[:,1].flatten(),color='blue',label='ML')
      plot(utest[1024:1024*2,3].flatten(),color='red',label='Truth')
      xlabel(r'$x$')
      ylabel(r'$\rho (x)$')
      savefig('rhou_test_depth' + str(depth) + '.png')

      figure(3)
      plot(yhat_train[:,2].flatten(),color='blue',label='ML')
      plot(utest[1024*2:1024*3,3].flatten(),color='red',label='Truth')
      xlabel(r'$x$')
      ylabel(r'$\rho(x)$')
      savefig('rhoE_test_depth' + str(depth) + '.png')

      figure(4)
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

axis_font = {'size':'24'}
figure(1)
for i in range(0,4):
  if (i == 0):
    plot(x[::4],U[i::4,0],color='black',label='Truth, training')
    plot(x[::4],yhat[i::4,0].detach().numpy(),color='blue',label='Prediction, training')
  else:
    plot(x[::4],U[i::4,0],color='black')
    plot(x[::4],yhat[i::4,0].detach().numpy(),color='blue')

legend(loc=1)
savefig('rho_train.png')

figure(2)
for i in range(0,25,4):
  if (i == 0):
    plot(x[::4],utest[:,i],color='black',label='Truth, testing')
    plot(x[::4],yhat_test[i::25,0].detach().numpy(),color='red',label='Prediction, testing')
  else:
    plot(x[::4],utest[:,i],color='black')
    plot(x[::4],yhat_test[i::25,0].detach().numpy(),color='red')
legend(loc=1)
savefig('rho_test.png')

xlabel(r'$x$',**axis_font)
ylabel(r'$\rho(x)$',**axis_font)
