from pylab import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn import DeepNN 
import time
import scipy.optimize
close("all")
torch.set_default_dtype(torch.float32)

#== Load data
data = np.load('snapshots_st_cn.npz')
U = data['snapshots'] 
mu1 = data['mu1']
mu2 = data['mu2']
x = data['x']
t = data['t']
xskip = 1
tskip = 10
x = x[::xskip]
t = t[::tskip]
U = U[::xskip,::tskip,:]
nx,nt,nparams= np.shape(U)
U = U.flatten(order='F')[:,None]

mu2v,mu1v,tv,xv = np.meshgrid(mu2,mu1,t,x)
xv = xv.flatten()
tv = tv.flatten()
mu1v = mu1v.flatten()
mu2v = mu2v.flatten()


input_features = np.append(xv[:,None],tv[:,None],axis=1)
input_features = np.append(input_features,mu1v[:,None],axis=1)
input_features = np.append(input_features,mu2v[:,None],axis=1)
ninputs = np.shape(input_features)[0]
#input_features_tensor = torch.tensor(input_features,dtype=torch.float32)
xtest = x*1.
ttest = t[32:33]
mu1test = mu1[2:3]
mu2test = mu2[2:3]

mu2vtest,mu1vtest,tvtest,xvtest = np.meshgrid(mu2test,mu1test,ttest,xtest)
xvtest = xvtest.flatten()
tvtest = tvtest.flatten()
mu1vtest = mu1vtest.flatten()
mu2vtest = mu2vtest.flatten()

input_features_test = np.append(xvtest[:,None],tvtest[:,None],axis=1)
input_features_test = np.append(input_features_test,mu1vtest[:,None],axis=1)
input_features_test = np.append(input_features_test,mu2vtest[:,None],axis=1)
Utest = data['snapshots'][::xskip,::tskip]
#===
depth=6
nbasis = 8
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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50000)

#Loss function

def loss_ls(weights):
  model.forward_list[-1].weight.data[0] = torch.tensor(weights,dtype=torch.float32)
  samples = np.array(range(0,ninputs),dtype='int')
  np.random.shuffle(samples)
  nsamples = 100000
  samples = samples[0:nsamples]
  yhat = model.forward(torch.tensor(input_features[samples],dtype=torch.float32))
  return U[samples].flatten() - yhat.detach().numpy().flatten()

def my_criterion(y,yhat):
  loss_mse = torch.mean( (y - yhat)**2 )
  return loss_mse 
#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


#Epochs
n_epochs = 20000
train_loss_hist = np.zeros(0)
t0 = time.time()
plot_freq = 10
ls_freq = 5
for epoch in range(1, n_epochs+1):
    if (epoch == 2000):
      optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    if (epoch == 3000):
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if (epoch == 4000):
      optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # monitor training loss
    train_loss = 0.0
    #Training
    for data in train_loader:
        data_d = data.to(device,dtype=torch.float32)
        inputs = data_d[:,0:4]
        y = data_d[:,4::]
        optimizer.zero_grad()
        yhat = model(inputs)
        loss = my_criterion(y,yhat)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*inputs.size(0)

    if (epoch%ls_freq == 0):
      weights0 = model.forward_list[-1].weight.data.detach().numpy().flatten()
      weights = scipy.optimize.least_squares(loss_ls,weights0,verbose=2).x
      model.forward_list[-1].weight.data[0] = torch.tensor(weights,dtype=torch.float32)

    train_loss = train_loss#/len(train_loader)
    train_loss_hist = np.append(train_loss_hist,train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    print('Time: {:.6f}'.format(time.time() - t0))

    if (epoch%plot_freq == 0):        
      figure(1)
      yhat_train = model.forward(torch.Tensor(input_features_test)).detach().numpy()
      plot(yhat_train[:,0].flatten(),color='blue',label='ML')
      plot(Utest[:,32,10].flatten(),color='red',label='Truth')
      xlabel(r'$x$')
      ylabel(r'$\rho(x)$')
      savefig('u_test_depth' + str(depth) + '.png')
      figure(2)
      loglog(train_loss_hist)
      xlabel(r'Epoch')
      ylabel(r'Loss')
      savefig('train_loss_depth' + str(depth) + '.png')
      close("all")

      torch.save(model.state_dict(), 'tmp_model_depth' + str(depth))

  
tf = time.time()

'''

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
'''
