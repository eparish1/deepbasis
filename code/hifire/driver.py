from pylab import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn_multistate import DeepNN 
import time
close("all")
torch.set_default_dtype(torch.float32)

def trainModel(modelName,Normalize=False,BoxInit=False):
    #== Load data
    data = np.load('training_data.npz')
    U = data['U'] 
    if Normalize:
      x = (data['X'][:,0] - np.amin(data['X'][:,0]))/( np.amax(data['X'][:,0] - np.amin(data['X'][:,0])) )
      y = (data['X'][:,1] - np.amin(data['X'][:,1]))/( np.amax(data['X'][:,1] - np.amin(data['X'][:,1])) )
      mu1 = (data['params'][:,0] - np.amin(data['params'][:,0]))/( np.amax(data['params'][:,0]) - np.amin(data['params'][:,0]))
      mu2 = (data['params'][:,1] - np.amin(data['params'][:,1]))/( np.amax(data['params'][:,1]) - np.amin(data['params'][:,1]))

      for i in range(0,6):
        U[:,i::6] = (U[:,i::6] - np.amin(U[:,i::6]) ) / ( np.amax(U[:,i::6]) - np.amin(U[:,i::6]))

    else:
      x = data['X'][:,0]
      y = data['X'][:,1]
      mu1 = data['params'][:,0]
      mu2 = data['params'][:,1]

    nparams,N =  np.shape(U)
    nc = 6
    U2 = np.zeros((6,nparams,int(N/6)))
    for i in range(0,6):
      U2[i,:,:] = U[:,i::6]
    U = np.reshape(U2,(6,nparams*int(N/6))) #will run through all x, then params
    U = np.rollaxis(U,1)
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
    input_features_indices = np.array(range(0,input_features_size),dtype='int')
    np.random.shuffle(input_features_indices)
    fraction_to_retain = 0.5 
    input_features_indices = input_features_indices[0:int(fraction_to_retain*input_features_size)]
    print(np.shape(input_features))
    print(np.shape(U),np.shape(input_features_indices)) 
    input_features = input_features[input_features_indices]
    U = U[input_features_indices] 
    #===
    model = DeepNN(depth,nbasis,nc,BoxInit)
    ## This bit of code can load a model if we want to resume training
    #try:
    #model.load_state_dict(torch.load('tmp_model_depth' + str(depth),map_location='cuda'))
    #model.load_state_dict(torch.load('tmp_model_depth' + str(depth) + '_nbasis_' + str(nbasis),map_location='cuda'))
    #  print('Succesfully loaded model!')
    #except:
    #  print('Failed to load model')

    device = 'cpu'
    model.to(device)


    train_data = np.float32(np.append(input_features,U,axis=1))
    print('Training data shape = ' + str(np.shape(train_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=10000)

    full_input = torch.tensor(train_data,dtype=torch.float32)

    #Loss function
    def my_criterion(y,yhat):
      loss_mse = torch.mean( (y - yhat[:,:,0])**2 )
      return loss_mse 
    #Optimizer
    learning_rate = 1e-2
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
    n_its_until_possible_reset = 0
    for epoch in range(1, n_epochs+1):
        if epoch>11:
            if np.mean(train_loss_hist[-10::]) > np.mean(train_loss_hist[-20:-10]) and learning_rate > 1e-5 and n_its_until_possible_reset > 10:
                learning_rate = learning_rate*0.95
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print('Adjusting learning rate to ' + str(learning_rate))
                n_its_until_possible_reset = 0
#        if (epoch == 1000):
#          optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
#        if (epoch == 3000):
#          optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#        if (epoch == 7000):
#          optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
#        if (epoch == 9000):
#          optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
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

        train_loss = train_loss#/len(train_loader)
        train_loss_hist = np.append(train_loss_hist,train_loss)
        n_its_until_possible_reset += 1
        #print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}  \tSVD Loss: {:.6f}'.format(epoch, train_loss,test_loss,svd_loss))
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))
        print('Time: {:.6f}'.format(time.time() - t0))

        if (epoch%plot_freq == 0):       
          '''
          #This is if we want to plot stuff
          figure(1)
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

depth_a = np.array([7])
nbasis_a = np.array([8,16,24])
nmodels = 3
NormalizeData = True
BoxInit = False
for i in range(0,nmodels):
 for nbasis in nbasis_a:
  for depth in depth_a:
    print("=======================================================")
    print("=======================================================")
    print("Training the " + str(i) + " model with basis dimension " + str(nbasis))
    print("=======================================================")
    print("=======================================================")

    modelName = 'depth_' + str(depth) + '_nbasis_' + str(nbasis) + '_no_' + str(i) 
    trainModel(modelName,NormalizeData,BoxInit)
