from pylab import *
import numpy as np
from scipy.interpolate import lagrange
import scipy.optimize
from scipy.sparse.linalg import LinearOperator
from q1de import *
from deep_nn import DeepNN
import torch
import scipy.optimize
import time
close("all")
torch.set_default_dtype(torch.float32)

try: 
  from adolc import * 
except:
#  if (MPI.COMM_WORLD.Get_rank() == 0):
    print("adolc not found, can't use adolc automatic differentiation")


#global parameters
gamma = 1.4

N = 1024
x = np.linspace(0,10.,N)
dx = x[1]


depth=4
nbasis=4
models = [None]*3
models_str = ['rho','rhoU','rhoE']
for i in range(0,3):
  models[i] = DeepNN(depth,nbasis)
  np.random.seed(1)
  models[i].load_state_dict(torch.load(models_str[i] + '_model_depth' + str(depth),map_location='cpu'))

def computeVelocity2(u,mu):
  x_lagrange_points = np.array([0,5.,10.])
  A_lagrange_points = np.array([3.,mu,3.])
  poly = lagrange(x_lagrange_points,A_lagrange_points)
  dlnAdx = 1./poly(x)*poly.deriv(1)(x)
  return computeVelocity(u,dlnAdx,dx)

def solveSystem(mu,Phi):
  u = np.zeros((3,N)) 
  u[0] = 1.
  u[1] = 0.
  u[2] = 1./(gamma - 1.) + 0.5*u[0]*u[1]**2
  ## area profile
  x_lagrange_points = np.array([0,5.,10.])
  A_lagrange_points = np.array([3.,mu,3.])
  poly = lagrange(x_lagrange_points,A_lagrange_points)
  dlnAdx = 1./poly(x)*poly.deriv(1)(x)
  # construct local compute velocity function with specific area profile 
  def computeVelocityLocal(u):
    return computeVelocity(u,dlnAdx,dx)

  def residual(xhat):
    lam = 0.25 
    xhat = np.reshape(xhat,(3,nbasis))
    u = np.zeros((3,N))
    for i in range(0,3):
      u[i] = np.dot(Phi[i],xhat[i])
    resid = computeVelocityLocal(u) 
    return diff(resid.flatten(order='F'))# + lam*norm(xhat - xhat0)

  def residualGalerkin(xhat):
    lam = 0.1
    xhat = np.reshape(xhat,(3,nbasis))
    u = np.zeros((3,N))
    for i in range(0,3):
      u[i] = np.dot(Phi[i],xhat[i])
    resid = computeVelocityLocal(u)
    residOrtho = np.zeros((3,N))
    eps = 1e-3
    tau = -0.01
    for i in range(0,3):
      residOrtho[i] = resid[i*N:(i+1)*N] - np.dot(Phi[i],np.dot(Phi[i].transpose(),resid[i*N:(i+1)*N]))

    closure = tau/eps*(computeVelocityLocal(u + eps*residOrtho) - resid) 
    residG = np.zeros((3,nbasis))
    for i in range(0,3):
      residG[i] = np.dot(Phi[i].transpose(),resid[i*N:(i+1)*N] + closure[i*N:(i+1)*N])
    return residG.flatten() #+ lam*(xhat - xhat0)

  #xhat = scipy.optimize.least_squares(residual,xhat0.flatten(),verbose=2).x
  xhat = scipy.optimize.newton_krylov(residualGalerkin,xhat0.flatten(),verbose=2)
  u = np.zeros(3*N)
  xhat = np.reshape(xhat,(3,nbasis))
  for i in range(0,3):
    u[i*N:(i+1)*N] = np.dot(Phi[i],xhat[i])
  return u

data = np.load('../snapshots_vfine.npz')
#params = np.array([0.5,0.875,1.25,1.625])
params = np.linspace(0.5,1.625,25)
snapshots = np.zeros((3*N,np.size(params)))
snapshotsml = np.zeros((3*N,np.size(params)))

u = np.zeros((3,N))
for i in range(np.size(params)):

  xi,mui = np.meshgrid(x/10.,params[i:i+1])
  xi = xi.flatten()
  mui = mui.flatten() 
  input_features = np.append(xi[:,None],mui[:,None],axis=1)
  input_features2 = np.append(xi[:,None]+0.2,mui[:,None] ,axis=1)

  Phi = [None]*3
  for k in range(0,3):
    Phi[k] = models[k].createBasis(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
    #tmp2 = models[k].createBasis(torch.tensor(input_features2,dtype=torch.float32)).detach().numpy()
    #Phi[k] = np.append(Phi[k],tmp2,axis=1)
    Phi[k],s,_ = np.linalg.svd(Phi[k],full_matrices=False)
  xhat0 = np.zeros((3,nbasis))
  uic = data['snapshots'][:,i]
  uml = np.zeros((3,N))
  for k in range(0,3):
    uml[k] = models[k].forward(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()[:,0]
  up = np.zeros((3,N))
  for k in range(0,3):
    xhat0[k] = np.dot(Phi[k].transpose(),uic[k*N:(k+1)*N])
    up[k] = np.dot(Phi[k],xhat0[k])
  print('On parameter ' + str(i) + ' of ' + str(np.size(params)))
  snapshots[:,i] = solveSystem(params[i],Phi)
  snapshotsml[:,i] = uml.flatten() 

