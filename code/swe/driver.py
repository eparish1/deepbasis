from pylab import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn import DeepNN 
import time
close("all")
torch.set_default_dtype(torch.float32)
solData = np.load('swe_sols_data_U.npz')
grid = np.load('DGgrid_block0.npz')
parameter_vec_g = solData['parameter_vec_g'] 
parameter_vec_f = solData['parameter_vec_f']
def trainModel(modelName,Normalize=False,BoxInit=False):
    #== Load data
    U = solData['U'] 
    if Normalize:
      mu1 = (parameter_vec_g - np.amin(parameter_vec_g))/( np.amax(parameter_vec_g) - np.amin(parameter_vec_g))
      mu2 = (parameter_vec_f - np.amin(parameter_vec_f))/( np.amax(parameter_vec_f) - np.amin(parameter_vec_f))
      x = grid['x']/np.amax(grid['x'])
      y = grid['y']/np.amax(grid['y'])
      t = solData['t']/np.amax(solData['t'])
    else:
      mu1 = parameter_vec_g
      mu2 = parameter_vec_f
      x = grid['x']
      y = grid['y']
      t = solData['t']

    xskip = 1
    yskip = 1
    tskip = 1
    x = x[::xskip,0].flatten()
    y = y[0,::yskip].flatten()
    t = t[::tskip].flatten()
    U = U[:,::tskip,:,::xskip,::yskip,0]
    U = np.rollaxis(U,2,0) #
    nconserved, nparams , nt , nx , ny= np.shape(U)
    U = np.reshape(U,(nconserved,nparams*nt*nx*ny),order='C') #parameter_vec_f sols are stored sequentially

    #mu1v,mu2v = np.meshgrid(mu1v,mu2v)
    #mu1v = mu1v.flatten()
    #mu2v = mu2v.flatten()
    #mu1v = np.repeat(mu1v,nx*nt*ny)
    #mu2v = np.repeat(mu1v,nx*nt*ny)

    #yvl,xvl,tvl = np.meshgrid(y,x,t)

    #t2,x2,y2 = np.meshgrid(t,x,y,indexing='ij')
    #x2 = x2.flatten()
    #y2 = y2.flatten()
    #t2 = t2.flatten()

    tvl,xvl,yvl = np.meshgrid(t,x,y,indexing='ij')
    xvl = xvl.flatten()
    yvl = yvl.flatten()
    tvl = tvl.flatten()
    nl = np.size(xvl)
    xv = np.zeros(nl*nparams)
    yv = np.zeros(nl*nparams)
    tv = np.zeros(nl*nparams)
    mu1v = np.zeros(nl*nparams)
    mu2v = np.zeros(nl*nparams)
    for i in range(0,nparams):
      xv[i*nl:(i+1)*nl] = xvl
      yv[i*nl:(i+1)*nl] = yvl
      tv[i*nl:(i+1)*nl] = tvl
      mu1v[i*nl:(i+1)*nl] = mu1[i] 
      mu2v[i*nl:(i+1)*nl] = mu2[i]


    input_features = np.append(tv[:,None],xv[:,None],axis=1)
    input_features = np.append(input_features,yv[:,None],axis=1)
    input_features = np.append(input_features,mu1v[:,None],axis=1)
    input_features = np.append(input_features,mu2v[:,None],axis=1)
    ninput = np.size(input_features[:,0])
    indices = np.array(range(0,ninput),dtype='int')
    choices = np.random.choice(indices,int(ninput*0.05),replace=False)
    input_features = input_features[choices,:]
    train_data = np.float32(np.append(input_features,np.rollaxis(U[:,choices],1),axis=1))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=5000)

    tstep = 75
    input_features_test = np.append(tv[nx*ny*tstep:nx*ny*(tstep+1),None],xv[nx*ny*tstep:nx*ny*(tstep+1),None],axis=1)
    input_features_test = np.append(input_features_test,yv[nx*ny*tstep:nx*ny*(tstep+1),None],axis=1)
    input_features_test = np.append(input_features_test,mu1v[tstep*nx*ny:nx*ny*(tstep+1),None],axis=1)
    input_features_test = np.append(input_features_test,mu2v[tstep*nx*ny:nx*ny*(tstep+1),None],axis=1)
    test_data = np.float32(np.append(input_features_test,np.rollaxis(U[:,nx*ny*tstep:nx*ny*(tstep+1)],1),axis=1))
    test_data_tensor = torch.tensor(test_data,dtype=torch.float32)

    model = DeepNN(depth,nbasis,3,BoxInit)
    model.load_state_dict(torch.load('depth_' + str(depth) + '_nbasis_' + str(nbasis) + '_no_0_trained',map_location='cpu'))
    device = 'cpu'
    model.to(device)



    full_input = torch.tensor(train_data,dtype=torch.float32)
    #test_data = np.float32(np.append(input_features_test,Utest,axis=1))
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=int(np.size(test_data)))

    #Loss function
    def my_criterion(y,yhat):
      loss_mse = torch.mean( (y - yhat[:,:,0])**2 )
      return loss_mse 
    #Optimizer
    learning_rate = 5e-9
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
          optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        if (epoch == 2000):
          optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            inputs = data_d[:,0:5]
            y = data_d[:,5::]
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = my_criterion(y,yhat)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()#*inputs.size(0)

        train_loss = train_loss#/len(train_loader)
        train_loss_hist = np.append(train_loss_hist,train_loss)

        yhat_test = model.forward(test_data_tensor[:,0:5]).detach().numpy()
        y_test =np.rollaxis(np.reshape( test_data_tensor[:,5::].detach().numpy() , (nx,ny,3)) ,2,0)
             
        yhat_test = np.rollaxis(np.reshape(yhat_test,(nx,ny,3)),2,0)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        print('Time: {:.6f}'.format(time.time() - t0))
        fig,ax = plt.subplots()
        if (epoch%plot_freq == 0):     
          f,(ax1,ax2) = plt.subplots(1,2) 
          ax1.contourf(yhat_test[0],100)
          ax2.contourf(y_test[0],100)
          savefig('utraining.png')
          close("all")

          plot(diag(yhat_test[0]),'--')
          plot(diag(y_test[0]))
          savefig('utraining_2.png')
          close("all")

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
nbasis = 10 
nmodels = 5
NormalizeData = True
BoxInit = False
for i in range(0,nmodels):
  print("=======================================================")
  print("=======================================================")
  print("Training the " + str(i) + " model ")
  print("=======================================================")
  print("=======================================================")

  modelName = 'depth_' + str(depth) + '_nbasis_' + str(nbasis) + '_no_' + str(i) 
  trainModel(modelName,NormalizeData,BoxInit)
