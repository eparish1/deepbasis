from pylab import *
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from deep_nn import DeepNN 

import numpy as np
from scipy.interpolate import lagrange
import scipy.optimize
from scipy.sparse.linalg import LinearOperator
from q1de import *
try: 
  from adolc import * 
except:
#  if (MPI.COMM_WORLD.Get_rank() == 0):
    print("adolc not found, can't use adolc automatic differentiation")

depth=4
model = DeepNN(depth)
np.random.seed(1)
model.load_state_dict(torch.load('../tmp_model_depth' + str(depth),map_location='cpu'))

#global parameters
gamma = 1.4

N = 1024
x = np.linspace(0,10.,N)
dx = x[1]


weights_0 = model.forward_list[0].weight.data*1.
K = np.prod(weights_0.shape)
samples = np.array(range(0,1024*3))
nsamples = 500
np.random.shuffle(samples)
samples = samples[0:nsamples]

Phi = np.zeros((N,K))
for i in range(0,K):
  Phi[:,i] = scipy.special.eval_chebyt(i, x)

def solveSystem(mu):
  model.forward_list[0].weight.data = weights_0
  u = np.zeros((3,N)) 
  u[0] = 1.
  u[1] = 0.
  u[2] = 1./(gamma - 1.) + 0.5*u[0]*u[1]**2
  inputs = np.zeros((N,2))
  inputs[:,0] = x/10.
  inputs[:,1] = mu
  inputs = torch.tensor(inputs)
  u_ml = np.rollaxis( model.forward(inputs).detach().numpy(),1)

  ## area profile
  x_lagrange_points = np.array([0,5.,10.])
  A_lagrange_points = np.array([3.,mu,3.])
  poly = lagrange(x_lagrange_points,A_lagrange_points)
  dlnAdx = 1./poly(x)*poly.deriv(1)(x)
  # construct local compute velocity function with specific area profile 
  #def jacobian(xhat):
  #  #xhat = np.reshape(xhat,(4,2))
  #  J = np.zeros((N*3,np.size(xhat)))
  #  model.forward_list[0].weight.data = weights_0 + torch.tensor(np.reshape( xhat,(4,2)) )
  #  u0 = np.rollaxis( model.forward(inputs).detach().numpy() , 1).flatten()
  #  eps = 1e-2
  #  for i in range(0,np.size(xhat)):
  #    xhat_l = xhat*1.
  #    xhat_l[i] += eps
  #    xhat_l = np.reshape(xhat_l,(4,2))
  #    model.forward_list[0].weight.data = weights_0 + torch.tensor(xhat_l)
  #    u = np.rollaxis( model.forward(inputs).detach().numpy() , 1).flatten()
  #    J[:,i] = (u - u0)/eps 
  #  return J
 
  def computeVelocityLocal(u):
    return computeVelocity(u,dlnAdx,dx)
  def residual(xhat):
    lam = 5.
    xhat = np.reshape(xhat,weights_0.shape)
    model.forward_list[0].weight.data = weights_0 + torch.tensor(xhat)
    u = np.rollaxis( model.forward(inputs).detach().numpy(),1)
    residual = computeVelocityLocal(u)
    #print(np.shape(residual),np.shape(u))
    return np.sqrt(np.abs(residual.flatten())) + lam*(u - u_ml).flatten()

  def residual_galerkin(xhat):
    xhat = np.reshape(xhat,weights_0.shape)
    model.forward_list[0].weight.data = weights_0 + torch.tensor(xhat)
    u = np.rollaxis( model.forward(inputs).detach().numpy(),1)
    residual = computeVelocityLocal(u)
    #J = jacobian(xhat.flatten())
    return np.dot(Phi.transpose(),residual[0:1024].flatten())

 
  t = 0
  dt = 5.e-3
  et = 100

  counter = 0
  # do pesudo time stepping to get a good guess 
  xhat = np.zeros(K)
  xhat = scipy.optimize.least_squares(residual_galerkin,xhat,verbose=2,ftol=1e-6,xtol=1e-6).x
  #xhat = scipy.optimize.newton_krylov(residual_galerkin,xhat,verbose=4)

  u = np.rollaxis( model.forward(inputs).detach().numpy(),1)
  #print('Final residual norm = ' + str(fnorm))
  # run through a final newton krylov step to get converged solution
  #u = scipy.optimize.newton_krylov(computeVelocityLocal,u.flatten(),verbose=4)
  return u.flatten(),u_ml.flatten()

#params = np.array([0.5,0.875,1.25,1.625])
params = np.linspace(0.5,1.625,25)
snapshots = np.zeros((3*N,np.size(params)))
snapshots_ml  = np.zeros((3*N,np.size(params)))

u = np.zeros((3,N))
for i in range(np.size(params)):
  print('On parameter ' + str(i) + ' of ' + str(np.size(params)))
  snapshots[:,i],snapshots_ml[:,i] = solveSystem(params[i])
   

np.savez('snapshots_vfine_rom',snapshots=snapshots,snapshots_ml=snapshots_ml,params=params,x=x)
snapshots_fom = load('../../snapshots_vfine.npz')['snapshots']
print('Error = ' + str(np.linalg.norm(snapshots - snapshots_fom)))
print('ML Error = ' + str(np.linalg.norm(snapshots_ml - snapshots_fom)))

