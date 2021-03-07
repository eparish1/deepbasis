from pylab import *
import scipy.optimize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn import DeepNN
import scipy.optimize
import time
close("all")
torch.set_default_dtype(torch.float32)

def f(u):
  ux = np.zeros(np.size(u))
  ux[1::] = 0.5/dx*(u[1::]**2 - u[0:-1]**2)
  ux[0] = 0.5/dx*(u[0]**2 - mu1**2)
  return -ux + 0.02*exp(mu2*x)

def spaceTimeResidual(u):
  u = np.reshape(u,(nx,nt))
  residual = np.zeros((nx,nt))
  u0 = uIC*1.
  for i in range(0,nt):
    local_residual = u[:,i] - u0 - 0.5*dt*(f(u[:,i]) + f(u0))
    residual[:,i] = local_residual
    u0 = u[:,i]*1.
  return residual.flatten()

def residual(u):
  return u - un - 0.5*dt*( f(u) + f(un) )


def spaceTimeResidualLsRom(xhat):
  uml = np.dot(Phi,xhat)
  uml = np.reshape(uml,(nx,nt),order='F')
  residual = spaceTimeResidual(uml)
  return np.sqrt(np.abs(residual.flatten()))


def spaceTimeResidualGalerkinRom(xhat):
  uml = np.dot(Phi,xhat)
  uml = np.reshape(uml,(nx,nt),order='F')
  residual = spaceTimeResidual(uml)
  return np.dot(Phi.transpose(),residual)  
nx = 256
x = linspace(0,100,nx)
dx = x[1]
uIC = np.ones(nx)
et = 35 
mu1 = 4.5
mu2 = 0.03#25
mu1a = np.array([mu1])
mu2a = np.array([mu2])

save_freq = 1
dx = x[1]
t = 0
u = uIC*1.
dt = 0.07
usavel = np.zeros((nx,0))
counter =0
tsave = np.zeros(0)

#===
depth=6
nbasis=8
model = DeepNN(depth,nbasis)
np.random.seed(1)
try:
  model.load_state_dict(torch.load('tmp_model_depth' + str(depth),map_location='cpu'))
  print('Succesfully loaded model!')
except:
  print('Failed to load model')
data = np.load('../snapshots_st_cn.npz')
ta = data['t']
nt = np.size(ta)
mu2vtest,mu1vtest,tvtest,xvtest = np.meshgrid(mu2a,mu1a,ta,x)
xvtest = xvtest.flatten()
tvtest = tvtest.flatten()
mu1vtest = mu1vtest.flatten()
mu2vtest = mu2vtest.flatten()

input_features = np.append(xvtest[:,None],tvtest[:,None],axis=1)
input_features = np.append(input_features,mu1vtest[:,None],axis=1)
input_features = np.append(input_features,mu2vtest[:,None],axis=1)

device = 'cpu'
model.to(device)
Phi = model.createBasis(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
Phi,s,_ = np.linalg.svd(Phi,full_matrices=False)
uml = model.forward(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()[:,0]
xhat0 = np.dot(Phi.transpose(),uml).flatten()
xhat = scipy.optimize.newton_krylov(spaceTimeResidualGalerkinRom,xhat0,verbose=4)
xhat_ls = scipy.optimize.least_squares(spaceTimeResidualLsRom,xhat0,verbose=2).x
urom = np.dot(Phi,xhat)
urom = np.reshape(urom,(nx,nt),order='F')

urom_ls = np.dot(Phi,xhat_ls)
urom_ls = np.reshape(urom_ls,(nx,nt),order='F')


uml = np.reshape(uml,(nx,nt),order='F')

t = 0
u = uIC*1.
dt = 0.07
ufom = np.zeros((nx,0))
counter =0
tsave = np.zeros(0)
while (t <= et):
  un = u*1.
  u = scipy.optimize.newton_krylov(residual,un,verbose=4)
  t += dt
  if (counter  % save_freq == 0):
    ufom = np.append(ufom,u[:,None],axis=1)
    tsave = np.append(tsave,t)
  counter += 1


#fom = np.load('../snapshots_st_cn_test.npz')
#ufom = fom['snapshots'][:,:,0]
'''
uml = np.reshape(uml,(nt,nx))
Phi,_,_ = np.linalg.svd(Phi,full_matrices=False)

#while (t <= et):
#  un = u*1.
#  u = scipy.optimize.newton_krylov(spaceTimeResidual,un,verbose=4)
  #for i in range(0,4):
  #  u = u0 + dt*rk4const[i]*f(u)
  t += dt
  if (counter  % save_freq == 0):
    usavel = np.append(usavel,u[:,None],axis=1)
    tsave = np.append(tsave,t)
  counter += 1

if k == 0:
  usave = usavel[:,:,None]*1.
else:
  usave = np.append(usave,usavel[:,:,None],axis=2)





'''



