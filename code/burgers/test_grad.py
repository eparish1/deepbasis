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
data = np.load('snapshots_st_cn.npz')
ta = data['t']
nt = np.size(ta)
mu2vtest,mu1vtest,tvtest,xvtest = np.meshgrid(mu2a,mu1a,ta,x)
xvtest = xvtest.flatten()
tvtest = tvtest.flatten()
mu1vtest = mu1vtest.flatten()
mu2vtest = mu2vtest.flatten()

eps = 1e-3
input_features = np.append(xvtest[:,None],tvtest[:,None],axis=1)
input_features = np.append(input_features,mu1vtest[:,None],axis=1)
input_features = np.append(input_features,mu2vtest[:,None],axis=1)

input_features2 = np.append(xvtest[:,None],tvtest[:,None] + eps,axis=1)
input_features2 = np.append(input_features2,mu1vtest[:,None],axis=1)
input_features2 = np.append(input_features2,mu2vtest[:,None],axis=1)


device = 'cpu'
model.to(device)
Phi = model.createBasis(torch.tensor(input_features,dtype=torch.float32))

grad_t = np.zeros(np.shape(Phi))
if2t = torch.tensor(input_features2,dtype=torch.float32,requires_grad=True)
for i in range(0,8):
  Phi2 = model.createBasis(if2t)
  Phi2[:,i].backward(gradient=torch.ones_like(Phi2[:,i]),retain_graph=True)
  grad_t[:,i] = if2t.grad[:,1].detach().numpy()
  if2t.grad.data.zero_()

grad_t2 = ( (Phi2 - Phi)/eps ).detach().numpy()





#Phi,s,_ = np.linalg.svd(Phi,full_matrices=False0)

