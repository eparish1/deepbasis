import numpy as np
import torch
import torch.nn as nn
from tools import *
import sys
from deep_nn import *
import matplotlib.pyplot as plt
import time
from itertools import product

def yk_createInputData(data):
    mu2_m, mu1_m, t_m, x_m = np.meshgrid(data.mu2norm.mat, data.mu1norm.mat,\
                                          data.tnorm.mat, data.xnorm.mat)

    #Create the semi form of the input matrix minus the snapshots part for NN
    in_feat = np.append(x_m.flatten()[:, None], t_m.flatten()[:, None], axis=1)
    in_feat = np.append(in_feat, mu1_m.flatten()[:, None], axis=1)
    return np.append(in_feat, mu2_m.flatten()[:, None], axis=1)


def yk_createTrainParameter(data):
    #Normalize parameters 1
    data.mu1norm = data.createDataMat()
    data.mu1norm.normalize(data.params[0])

    #Normalize parameters 2
    data.mu2norm = data.createDataMat()
    data.mu2norm.normalize(data.params[1])

    #Normalize time
    data.tnorm = data.createDataMat()
    data.tnorm.normalize(data.t)

    #Normalize space
    data.xnorm = data.createDataMat()
    data.xnorm.normalize(data.x)

    return yk_createInputData(data)


def yk_createTrainData(modelName, data, nValidation):
    #Normalize snapshots
    data.snapshotNorm = data.createDataMat()
    data.snapshotNorm.normalize(data.snapshot)

    #Create the final training data
    data.train = data.createTrainMat()

    in_feat = yk_createTrainParameter(data)

    data.snapshotNorm.mat = data.snapshotNorm.mat.flatten(order='F')[:, None]
    data.meshdata = np.append(in_feat, data.snapshotNorm.mat, axis=1)

    #Create the actual training data matrix
    data.train.createData(modelName, data.meshdata, nValidation)

def yk_createValidateData(modelName, data, valPercent):
    data.validate = data.createValidateMat()
    data.validate.createData(modelName, data.meshdata, valPercent)

def yk_trainNeuralNetwork(data, modelName, depth, nbasis, n_epochs, BoxInit=False):
    learning_rate = 1e-2
    save_freq=1
    model = DeepNN(depth, nbasis, BoxInit)
    device = 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_criterion = nn.MSELoss()
    t0 = time.time()
    loss_train = []
    loss_val_vec = []
    for epoch in range(n_epochs):
        if epoch>2:
            if loss_train[epoch-1] > loss_train[epoch-2] and learning_rate > 1e-5:
                learning_rate = learning_rate*0.75
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # if (epoch == 8):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # if (epoch == 50):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        # if (epoch == 100):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # if (epoch == 150):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        model.train()
        loss = 0
        for data_batch_i in data.train.loader:
            data_device = data_batch_i.to(device)
            inputs = data_device[:,0:4]
            y = data_device[:,4::]
            optimizer.zero_grad()
            yhat = model(inputs)
            train_loss = my_criterion(y,yhat)
            train_loss.backward()
            optimizer.step()

            loss+=train_loss.item()
        loss = loss/len(data.train.loader)
        loss_train.append(loss)
        # if (epoch%save_freq == 0):
        torch.save(model.state_dict(), 'tmp' + modelName)
        np.savez('stats_' + modelName, train_loss=loss_train,walltime = \
                 time.time() - t0)

        if data.nValidation > 0:
            model.eval()
            loss_val = 0

            with torch.no_grad():
                for data_batch_i in data.validate.loader:
                    data_device = data_batch_i.to(device)
                    inputs_val = data_device[:,0:4]
                    y = data_device[:,4::]
                    yhat = model(inputs_val)
                    val_loss = my_criterion(y,yhat)
                    loss_val+=val_loss.item()
                loss_val = loss_val/len(data.validate.loader)
                loss_val_vec.append(loss_val)
            print("Time: {:.6f}, epoch : {}/{}, loss = {:.16f}, loss_val = {:.16f}".format(time.time() - t0, epoch + 1, n_epochs, loss, loss_val))

        else:
            print("Time: {:.6f}, epoch : {}/{}, loss = {:.16f}".format(time.time()-t0, epoch + 1, n_epochs, loss))

    torch.save(model.state_dict(), modelName + '_trained')
    np.savez('stats_' + modelName  , train_loss=loss_train,walltime = time\
.time() - t0)

    plt.figure(1)
    plt.plot(np.linspace(0, n_epochs, n_epochs), loss_train)
    plt.plot(np.linspace(0, n_epochs, n_epochs), loss_val_vec)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.savefig(modelName+'.pdf')
    plt.close()
    return model

def yk_postProcessTraining(model,modelName,data):
    device = 'cpu'
    testbs=data.train.mat.shape[0]
    tloader = torch.utils.data.DataLoader(data.train.mat, batch_size=testbs,shuffle=False)
    model.eval()
    with torch.no_grad():
        for test in tloader:
            data_d = test.to(device)
            inputs_test=data_d[:,0:4]
            y_test=data_d[:,4::]
            yhattest=model(inputs_test)
            Phi=model.createBasis(inputs_test)

            #plt.figure(2)

    Phi,_=np.linalg.qr(Phi, mode='reduced')
    PhiT = np.transpose(Phi)
    # print(PhiT[0:10,0:10])
    # print( data.train.mat[0:10,4])
    ufom_p = Phi @ (PhiT @ data.train.mat[:,4])
    # ufom_p = np.linalg.norm(Phi)
    #ufom_p = np.dot(np.transpose(Phi),  data.train.mat[:,4])
    #    ufom_p = np.dot(Phi,np.dot(PhiT, data.train.mat[:,4]))
    p_error=np.mean( (ufom_p-data.train.mat[:,4] )**2 )
    np.savez('train_projectedrrror_' + modelName  , mse=p_error)

def main(argv):
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_default_dtype(torch.float64)
    #--------------------------------------------------------------------------
    # ROM parameters and stuff
    #--------------------------------------------------------------------------
    data = np.load('snapshots_st_cn.npz')
    snapshot = data['snapshots']
    nx, nt, nmu = snapshot.shape
    t = data['t']
    x = data['x']
    params = [data['mu1'], data['mu2']]
    trainTotalPercent = 0.1
    valTotalPercent = 0.05
    nnval = int(valTotalPercent*nx*nt*nmu)
    nValidation = int((1.0-trainTotalPercent)*nx*nt*nmu)


    #--------------------------------------------------------------------------
    # Machine learning parameters and businesses go here sigh
    #--------------------------------------------------------------------------
    n_epochs = 500
    BATCH_SIZE = 1000
    bs=BATCH_SIZE
    depth_a = np.array([int(argv[0])])
    nbasis_a = np.array([int(argv[1])])
    nmodels = 1
    init = 13
    #--------------------------------------------------------------------------
    # NEURAL NETWORK BUSINESS GOES HERE
    #--------------------------------------------------------------------------
    #--------------------------------------------------------------------------
    # Machine Learning Section Training hereee
    #--------------------------------------------------------------------------
    for i in range(init, nmodels+init):
        for nbasis, depth in product(nbasis_a, depth_a):
            print("==========================================================")
            print("Training the " + str(i) + " model with basis dimension "\
                  + str(nbasis)+", depth "+ str(depth))
            print("==========================================================")
            modelName = 'dropout_depth_' + str(depth) + '_nbasis_' + \
                        str(nbasis) +'_no_' + str(i)

            #------------------------------------------------------------------
            # ROM parameters and stuff
            #------------------------------------------------------------------
            data = GroupMatrix(snapshot, params, t, x, BATCH_SIZE=bs)
            data.nValidation = nValidation
            yk_createTrainData(modelName, data, nValidation)

            print('Training data shape = ' + str(np.shape(data.train.mat)))
            #------------------------------------------------------------------
            # ROM parameters and stuff
            #------------------------------------------------------------------
            yk_createValidateData(modelName, data, nnval)
            print('Validating data shape = '+str(np.shape(data.validate.mat)))
            model=yk_trainNeuralNetwork(data, modelName,depth,nbasis, n_epochs)

            yk_postProcessTraining(model,modelName, data)

if __name__ == '__main__':
    main(sys.argv[1:])
