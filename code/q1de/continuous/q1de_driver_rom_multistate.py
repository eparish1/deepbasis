from pylab import *
import numpy as np
from scipy.interpolate import lagrange
import scipy.optimize
from scipy.sparse.linalg import LinearOperator
from q1de import *
from deep_nn_multistate import DeepNN
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
nx = N
x = np.linspace(0,10.,N)
dx = x[1]


depth=4
nbasis=8
model = DeepNN(depth,nbasis,3)
np.random.seed(1)
model.load_state_dict(torch.load('tmp_model_depth' + str(depth) + '_nbasis_' + str(nbasis),map_location='cpu'))

def computeVelocity2(u,mu):
  x_lagrange_points = np.array([0,5.,10.])
  A_lagrange_points = np.array([3.,mu,3.])
  poly = lagrange(x_lagrange_points,A_lagrange_points)
  dlnAdx = 1./poly(x)*poly.deriv(1)(x)
  r = computeVelocity(u,dlnAdx,dx,False)
  return r 

def solveSystem(mu,Phi,method='resmin'):
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
  def computeVelocityLocal(u,returnBC=True):
    return computeVelocity(u,dlnAdx,dx,returnBC)

  def residual(xhat):
    lam = 0.05 
    u = np.dot(Phi,xhat)
    u2 = np.reshape(u,(3,nx))
    resid,_,_=  computeVelocityLocal(u) 
    resid = np.reshape(resid,(3,nx))
    #resid[:,0] *= lam
    #resid[:,-1] *= lam
    return np.sqrt(np.abs(resid.flatten())) + lam*norm(xhat - xhat0) 

  def residualGalerkin(xhat):
    u = np.dot(Phi,xhat)
    resid,_,_ = computeVelocityLocal(u)
    return np.dot(Phi.transpose(),resid.flatten())

  def residualAPG(xhat):
    tau = -0.0005
    u = np.dot(Phi,xhat)
    eps = 1e-5
    resid,_,_ = computeVelocityLocal(u)
    resid_ortho = resid - np.dot(Phi,np.dot(Phi.transpose(),resid))
    PLQLu  = 1./eps*(computeVelocityLocal(u + eps*resid_ortho,False) - resid)

    return np.dot(Phi.transpose(),resid.flatten() + tau*PLQLu.flatten())


  if (method == "resmin"):
    xhat = scipy.optimize.least_squares(residual,xhat0.flatten(),verbose=2).x
  if (method == "Galerkin"):
    xhat = scipy.optimize.newton_krylov(residualGalerkin,xhat0.flatten(),verbose=2)
  u = np.dot(Phi,xhat)#np.zeros(3*N)
  return u

datarom = np.load('../snapshots.npz')
PhiRom,_,_ = np.linalg.svd(datarom['snapshots'],full_matrices=False)
data = np.load('../snapshots_extrapolate.npz')
snapshots_fom = data['snapshots']
#params = np.array([0.5,0.875,1.25,1.625])
#params = np.linspace(0.5,1.625,25)
params = np.linspace(0.2,2.0,10)
snapshots_resmin = np.zeros((3*N,np.size(params)))
snapshots_galerkin = np.zeros((3*N,np.size(params)))
snapshots_ml = np.zeros((3*N,np.size(params)))
residual_resmin = np.zeros(np.size(params))
residual_galerkin = np.zeros(np.size(params))
residual_ml = np.zeros(np.size(params))

u = np.zeros((3,N))

def createBasis(mu,nextra,dmu):
  xi,mui = np.meshgrid(x/10.,mu)
  xi = xi.flatten()
  mui = mui.flatten() 
  input_features = np.append(xi[:,None],mui[:,None],axis=1)
  Phi1 = model.createBasis(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
  Phi = np.zeros((3*nx,nbasis))
  for k in range(0,3):
    Phi[k*nx:(k+1)*nx] = Phi1[:,k]

  muextra = np.linspace(mu - dmu/2,mu+dmu/2,nextra)
  for j in range(0,np.size(muextra)):
    xi,mui = np.meshgrid(x/10.,muextra[j])
    xi = xi.flatten()
    mui = mui.flatten() 
    input_features = np.append(xi[:,None],mui[:,None],axis=1)
    Phi1 = model.createBasis(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
    Phi2 = np.zeros((3*nx,nbasis))
    for k in range(0,3):
      Phi2[k*nx:(k+1)*nx] = Phi1[:,k]
    Phi = np.append(Phi,Phi2,axis=1)
  Phi,s,v = np.linalg.svd(Phi,full_matrices=False)
  rel_energy = np.cumsum(s**2) / np.sum(s**2)
  K = np.size(rel_energy[rel_energy<0.999999])
  print('ROM Size = ' + str(K))
  Phi = Phi[:,0:K]
  return Phi
  
for i in range(np.size(params)):
#  Phi = createBasis(params[i],10,0.5)
  xi,mui = np.meshgrid(x/10.,params[i:i+1])
  xi = xi.flatten()
  mui = mui.flatten() 
  input_features = np.append(xi[:,None],mui[:,None],axis=1)
  Phi1 = model.createBasis(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
  Phi = np.zeros((3*nx,nbasis))
  for k in range(0,3):
    Phi[k*nx:(k+1)*nx] = Phi1[:,k]
#  Phi = np.append(Phi,PhiRom,axis=1)
  Phi,s,_ = np.linalg.svd(Phi,full_matrices=False)
  xhat0 = np.zeros(nbasis)
  uic = data['snapshots'][:,i]
  xhat0 = np.dot(Phi.transpose(),uic)
  uicp = np.dot(Phi,np.dot(Phi.transpose(),uic))
  uml = model.forward(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
  uml = np.rollaxis(uml,1).flatten()
#   xhat0[k] = np.dot(Phi[k].transpose(),uic[k*N:(k+1)*N])
#    up[k] = np.dot(Phi[k],xhat0[k])
#  print('On parameter ' + str(i) + ' of ' + str(np.size(params)))
  snapshots_resmin[:,i] = solveSystem(params[i],Phi,'resmin')
  residual_resmin[i] = np.linalg.norm(computeVelocity2(snapshots_resmin[:,i],params[i]))
  #snapshots_galerkin[:,i] = solveSystem(params[i],Phi,'Galerkin')
  #residual_galerkin[i] = np.linalg.norm(computeVelocity2(snapshots_galerkin[:,i],params[i]))
  snapshots_ml[:,i] = uml.flatten()
  residual_ml[i] = np.linalg.norm(computeVelocity2(snapshots_ml[:,i],params[i]))

 
print('Res Min Error  = ' + str(np.linalg.norm(snapshots_resmin - snapshots_fom)))
print('Galerkin Error = ' + str(np.linalg.norm(snapshots_galerkin - snapshots_fom)))
print('ML Error       = ' + str(np.linalg.norm(snapshots_ml - snapshots_fom)))

print('H1 Res Min Error  = ' + str(np.linalg.norm(diff(snapshots_resmin[0:1024] - snapshots_fom[0:1024]))))
print('H1 Galerkin Error = ' + str(np.linalg.norm(diff(snapshots_galerkin[0:1024] - snapshots_fom[0:1024]))))
print('H1 ML Error       = ' + str(np.linalg.norm(diff(snapshots_ml[0:1024] - snapshots_fom[0:1024]))))


print('Res Min Residual  = ' + str(np.linalg.norm(residual_resmin)))
print('Galerkin Residual = ' + str(np.linalg.norm(residual_galerkin)))
print('ML Residual       = ' + str(np.linalg.norm(residual_ml)))

