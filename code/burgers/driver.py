from pylab import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn import DeepNN 
import time
close("all")
torch.set_default_dtype(torch.float32)

def trainModel(modelName,Normalize=False,BoxInit=False):
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

    xtest = x*1.
    if Normalize:
      ttest = data['t'][255:256] / np.amax(data['t'])
    else:
      ttest = data['t'][255:256]

    mu1test = mu1[:]
    mu2test = mu2[:]

    mu2vtest,mu1vtest,tvtest,xvtest = np.meshgrid(mu2test,mu1test,ttest,xtest)
    xvtest = xvtest.flatten()
    tvtest = tvtest.flatten()
    mu1vtest = mu1vtest.flatten()
    mu2vtest = mu2vtest.flatten()

    input_features_test = np.append(xvtest[:,None],tvtest[:,None],axis=1)
    input_features_test = np.append(input_features_test,mu1vtest[:,None],axis=1)
    input_features_test = np.append(input_features_test,mu2vtest[:,None],axis=1)
    Utest = data['snapshots'][::xskip,255:256]
    Utest = Utest.flatten(order='F')[:,None]
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
    test_data = np.float32(np.append(input_features_test,Utest,axis=1))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=int(np.size(test_data)))

    #Loss function
    def my_criterion(y,yhat):
      loss_mse = torch.mean( (y - yhat)**2 )
      return loss_mse 
    #Optimizer
    learning_rate = 5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    #Epochs
    n_epochs = 20000
    train_loss_hist = np.zeros(0)
    test_loss_hist = np.zeros(0)

    t0 = time.time()
    plot_freq = 10
    save_freq = 100
    check_stopping_criterion = 100
    eye = torch.tensor(np.eye(nbasis),dtype=torch.float32)
    eye = eye.to(device,dtype=torch.float32)
    full_input_d = full_input.to(device,dtype=torch.float32)
    for epoch in range(1, n_epochs+1):
        if (epoch == 1000):
          optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        if (epoch == 2000):
          optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        if (epoch == 5000):
          optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        if (epoch == 10000):
          optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        if (epoch == 15000):
          optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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

        for data in test_loader:
            data_d = data.to(device,dtype=torch.float32)
            inputs = data_d[:,0:4]
            y = data_d[:,4::]
            yhat = model(inputs)
            test_loss = my_criterion(y,yhat).item()
        #print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}  \tSVD Loss: {:.6f}'.format(epoch, train_loss,test_loss,svd_loss))
        print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(epoch, train_loss,test_loss))
        print('Time: {:.6f}'.format(time.time() - t0))
        test_loss_hist = np.append(test_loss_hist,test_loss)

        if (epoch%plot_freq == 0):       
          '''
          figure(1)
          yhat_train = model.forward(torch.Tensor(input_features_test)).detach().numpy()
          plot(yhat_train[:,0].flatten(),color='blue',label='ML')
          plot(Utest[:,32,9].flatten(),color='red',label='Truth')
          xlabel(r'$x$')
          ylabel(r'$\rho(x)$')
          savefig('u_test_depth' + str(depth) + '.png')
          figure(2)
          loglog(train_loss_hist)
          xlabel(r'Epoch')
          ylabel(r'Loss')
          savefig('train_loss_depth' + str(depth) + '.png')
          close("all")
          '''
        if (epoch%save_freq == 0):       
          torch.save(model.state_dict(), 'tmp' + modelName)
          np.savez('stats_' + modelName, train_loss=train_loss_hist,test_loss=test_loss_hist,walltime = time.time() - t0)
        if (epoch%check_stopping_criterion == 0 and epoch > 200):      
            test_loss_average_past = np.mean(test_loss_hist[epoch - 200:: epoch - 100])
            test_loss_average_present = np.mean(test_loss_hist[epoch - 100::])
            print('=======================')
            print('Epoch: {} \tTest mean running average (-200 to -100 iterations): {:.6f} \tTesting mean running average (-100 to present): {:.6f}'.format(epoch, test_loss_average_past,test_loss_average_present))
            print('=======================')
            #if (test_loss_average_past < test_loss_average_present and learning_rate > 1e-6):
            #  print('Test loss no longer decreasing, lowering learning rate to' + str(learning_rate))
            #  learning_rate /= 2
            #  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            #elif (test_loss_average_past < test_loss_average_present and learning_rate <= 1e-6):
            #  print('Test loss no longer decreasing and minimum learning rate achieved, stopping training')
            #  epoch = 1e10


    torch.save(model.state_dict(), modelName + '_trained')
    np.savez('stats_' + modelName  , train_loss=train_loss_hist,test_loss=test_loss_hist,walltime = time.time() - t0)

depth=6
nbasis = 8
nmodels = 5
NormalizeData = False 
BoxInit = True
for i in range(0,nmodels):
  print("=======================================================")
  print("=======================================================")
  print("Training the " + str(i) + " model ")
  print("=======================================================")
  print("=======================================================")

  modelName = 'depth_' + str(depth) + '_nbasis_' + str(nbasis) + '_no_' + str(i) 
  trainModel(modelName,NormalizeData,BoxInit)
