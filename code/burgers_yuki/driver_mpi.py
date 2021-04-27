import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn import DeepNN 
import time
torch.set_default_dtype(torch.float32)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
Nranks = comm.Get_size()
def trainModel(modelName,nbasis,depth,Normalize=False,BoxInit=False):
    #== Load data
    data = np.load('snapshots_st_cn.npz')
    U = data['snapshots'] 
    if Normalize:
      mu1 = (data['mu1'] - np.amin(data['mu1']))/( np.amax(data['mu1']) - np.amin(data['mu1']))
      mu2 = (data['mu2'] - np.amin(data['mu2']))/( np.amax(data['mu2']) - np.amin(data['mu2']))
      x = data['x']/np.amax(data['x'])
      t = data['t']/np.amax(data['t'])
    else:
      mu1 = data['mu1']
      mu2 = data['mu2']
      x = data['x']
      t = data['t']

    xskip = 1
    tskip = 10
    x = x[::xskip]
    t = t[0:250:tskip]
    U = U[::xskip,0:250:tskip,:]
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

    #===
    model = DeepNN(depth,nbasis,BoxInit)
    #try:
    #model.load_state_dict(torch.load('tmp_model_depth' + str(depth),map_location='cuda'))
    #model.load_state_dict(torch.load('tmp_model_depth' + str(depth) + '_nbasis_' + str(nbasis),map_location='cuda'))
    #  print('Succesfully loaded model!')
    #except:
    #  print('Failed to load model')

    device = 'cpu'
    model.to(device)


    train_data = np.float32(np.append(input_features,U,axis=1))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10000)

    full_input = torch.tensor(train_data,dtype=torch.float32)

    #Loss function
    def my_criterion(y,yhat):
      loss_mse = torch.mean( (y - yhat)**2 )
      return loss_mse 
    #Optimizer
    learning_rate = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    #Epochs
    n_epochs = 10000
    train_loss_hist = np.zeros(0)

    t0 = time.time()
    plot_freq = 10
    save_freq = 100
    check_stopping_criterion = 100
    eye = torch.tensor(np.eye(nbasis),dtype=torch.float32)
    eye = eye.to(device,dtype=torch.float32)
    full_input_d = full_input.to(device,dtype=torch.float32)
    for epoch in range(1, n_epochs+1):
        if (epoch == 1000):
          print('Approximate walltime for training this model = ' + str( (time.time() -t0)*10./3600.) + ' hours ')
          optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        if (epoch == 3000):
          optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        if (epoch == 7000):
          optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        if (epoch == 9000):
          optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
#        if (epoch == 15000):
#          optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
            train_loss += loss.item()#*inputs.size(0)

        #Phi = model.createBasis(full_input[:,0:4].to(device,dtype=torch.float32))#.detach().numpy()
        #PhiTPhi = torch.matmul(Phi.transpose(1,0),Phi)
        #Phi,s,_ = torch.svd(Phi, compute_uv=True)  
        #svd_loss = torch.mean((PhiTPhi - eye)**2)#(s[0]/s[-1]).item()
        #svd_loss = (s[0]/s[-1])
        #loss = svd_loss
        #loss.backward()
        #optimizer.step()
        train_loss = train_loss#/len(train_loader)
        train_loss_hist = np.append(train_loss_hist,train_loss)

        #print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))
        print('Time: {:.6f}'.format(time.time() - t0))

        if (epoch%plot_freq == 0):       
          '''
          figure(2)
          loglog(train_loss_hist)
          xlabel(r'Epoch')
          ylabel(r'Loss')
          savefig('train_loss_depth' + str(depth) + '.png')
          close("all")
          '''
        if (epoch%save_freq == 0):       
          torch.save(model.state_dict(), 'tmp' + modelName)
          np.savez('stats_' + modelName, train_loss=train_loss_hist,walltime = time.time() - t0)


    torch.save(model.state_dict(), modelName + '_trained')
    np.savez('stats_' + modelName  , train_loss=train_loss_hist,walltime = time.time() - t0)

depth_a = np.array([5])#,4,5,6])
nbasis_a = np.array([1,3,5,10,15,20,25,30,35,40,45,50])
nmodels = 4
model_a = np.array(range(0,nmodels),dtype='int')
nbasis_a,depth_a,model_a = np.meshgrid(nbasis_a,depth_a,model_a)
nbasis_a = nbasis_a.flatten()
model_a = model_a.flatten()
depth_a = depth_a.flatten()
nruns = np.size(model_a)
NormalizeData = True
BoxInit = False
runs_per_rank = int(np.ceil(nruns*1./Nranks))
starting_rank = min(runs_per_rank * rank,nruns)
ending_rank = min( runs_per_rank*(rank+1),nruns)
print('Runs per rank = ' + str(runs_per_rank))
print('Rank ' + str(rank) + ' starting on ' + str(starting_rank))
print('Rank ' + str(rank) + ' ending on ' + str(ending_rank))
if (rank == 0):
  print('Runs per rank = ' + str(runs_per_rank))
for i in range(starting_rank,ending_rank):
  print("=======================================================")
  print("=======================================================")
  print("Training the " + str(model_a[i]) + " model with basis dimension " + str(nbasis_a[i]) + " and depth of " + str(depth_a[i]))
  print("=======================================================")
  print("=======================================================")
  nbasis = nbasis_a[i] 
  depth = depth_a[i]
  modelName = 'depth_' + str(depth_a[i]) + '_nbasis_' + str(nbasis) + '_no_' + str(model_a[i]) 
  t0 = time.time()
  trainModel(modelName,nbasis,depth,NormalizeData,BoxInit)
  wallTime = time.time() - t0
  print("On Rank " + str(rank) + " training took " + str(wallTime) + ' seconds ')

