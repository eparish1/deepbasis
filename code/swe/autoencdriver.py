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
from shallow_autoencoder import ShallowAutoencoder
import time
close("all")
torch.set_default_dtype(torch.float64)

#== Load data
#U = np.load('../snapshots_st_cn.npz')['snapshots'][:,:][:,::5,::1] #nx x nt x nparams
U = np.load('../swe_sols_data.npz')['U'][:,:,:,:,:,0] #nparams x  nt x 3 x  nx x ny x 1
nparams,nt,nchan,nx,ny = np.shape(U)

N = nchan*nx*ny
U = np.reshape(U,(nparams*nt,nchan*nx*ny))
U = U[:,None,:,None]
#U = np.rollaxis(np.rollaxis(U,2,0),2,1) #nparams x nt x nx
#U = np.reshape(U,(nparams*nt,nx))[:,None,:,None]
U_ref = 0.

Utest = U#np.load('../snapshots_st_cn.npz')['snapshots'][:,:][:,3::5,::1] #nx x nt x nparams
#nx,nt,nparams = np.shape(Utest)
#Utest = np.rollaxis(np.rollaxis(Utest,2,0),2,1) #nparams x nt x nx
#Utest = np.reshape(Utest,(nparams*nt,nx))[:,None,:,None]
#===

#== Set model arcitechture
num_latent_states_a = np.array([10])
depth_a = np.array([1])
for depth in depth_a:
 for num_latent_states in num_latent_states_a:
  Phi,_,_ = np.linalg.svd( np.rollaxis(U[:,0,:,0],1),full_matrices=False)
  Phi = Phi[:,0:num_latent_states]
  U_project = np.dot(Phi, np.dot( Phi.transpose(), np.rollaxis(U[:,0,:,0],1) - U_ref)) + U_ref
  U_project = np.rollaxis(U_project,1)[:,None,:,None]
  
  model = ShallowAutoencoder(1,N,num_latent_states,U[:],depth)
  
  np.random.seed(1)
  shuffle_a = np.array(range(0,np.shape(U)[0]),dtype='int')
  shuffle(shuffle_a)
  U = U[shuffle_a]
  U_project = U_project[shuffle_a]


  try:
    model.load_state_dict(torch.load('tmp_model_depth' + str(depth),map_location='cpu'))
    print('Succesfully loaded model!')
  except:
    print('Failed to load model')
  
  device = 'cpu'
  model.to(device)
  
  
  U_scale = 1.
  utrain = (U - U_ref)/U_scale
  utest = (Utest - U_ref)/U_scale
  train_loader = torch.utils.data.DataLoader(utrain, batch_size=100)
  
  #Loss function
  def my_criterion(outputs,inputs): 
    loss_mse = torch.mean( (outputs - inputs)**2 ) 
    return loss_mse
  #Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  
   #Epochs
  n_epochs = 10000
  #inputs = torch.as_tensor(utrain[:],dtype=torch.float)
  plot_freq = 1
  train_loss_hist = np.zeros(0)
  train_loss_hist_sparse = np.zeros(0)
  test_loss_hist_sparse = np.zeros(0)

  t0 = time.time()
  plot_freq = 10
  stopTraining = False
  for epoch in range(1, n_epochs+1):
      if (epoch == 1000):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
      if (epoch == 5000):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
 
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
  
   
      train_loss = train_loss#/len(train_loader)
      train_loss_hist = np.append(train_loss_hist,train_loss)
      print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
      print('Time: {:.6f}'.format(time.time() - t0))
  
      if (epoch%plot_freq == 0):        
        figure(1)
        yhat_train = model.forward(torch.Tensor(utrain)).detach().numpy() + U_ref
        yhat_test = model.forward(torch.Tensor(utest)).detach().numpy() + U_ref

        error_ml = np.mean( (yhat_train - U_ref - utrain )**2 )*np.shape(U)[0]
        error_ml_test = np.mean( (yhat_test - U_ref - utest )**2 )*np.shape(U)[0]
        error_pca = np.linalg.norm(U_project - (utrain + U_ref))

        train_loss_hist_sparse = np.append(train_loss_hist_sparse,error_ml)
        test_loss_hist_sparse = np.append(test_loss_hist_sparse,error_ml_test)

        print('ML error (training set) = ' + str(np.linalg.norm(error_ml)))
        print('ML error (testing set) = ' + str(np.linalg.norm(error_ml_test)))

        print('PCA error = ' + str(np.linalg.norm(error_pca)))
  
        plot(yhat_train[-1].flatten(),color='blue',label='Autoencoder')
        plot(utrain[-1,:,0:-1].flatten() + U_ref,'--',color='blue',label='Training')
        plot(U_project[-1,:].flatten(),'-.',color='red',label='Projection')
        legend(loc=1)
        savefig('training_results_depth' + str(depth) + '.png')
        
        figure(2)
        loglog(train_loss_hist)
        xlabel(r'Epoch')
        ylabel(r'Loss')
        savefig('train_loss_depth' + str(depth) + '.png')

        figure(3)
        epoch_sparse  = np.linspace(1,np.size(train_loss_hist_sparse),np.size(train_loss_hist_sparse))*10
        loglog(epoch_sparse,train_loss_hist_sparse,color='blue',label='Training')
        loglog(epoch_sparse,test_loss_hist_sparse,color='red',label='Testing')
        

        xlabel(r'Epoch')
        ylabel(r'Loss')
        legend(loc=1)
        savefig('train_test_loss_depth' + str(depth) + '.png')

        close("all")
        torch.save(model.state_dict(), 'tmp_model_depth' + str(depth))

  
  tf = time.time()
  np.savez('model_stats_depth' + str(depth),train_loss_hist=train_loss_hist,error_ml=error_ml,error_pca=error_pca,walltime=tf - t0)

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
