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
nbasis=2
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
  def residual(xhat,l1=True):
    lam = 0.05 
    resid = np.zeros(np.shape(Phi)[0])
    rg = np.zeros((0,3))
    u = np.dot(Phi,xhat)
    u2 = np.reshape(u,(np.size(xi),3),order='F')
    for i in range(0,np.size(params)):
      ul = np.rollaxis(u2[i*nx:(i+1)*nx],1)
      rl = computeVelocity2(ul,mu[i])
      rl = np.rollaxis( np.reshape(rl,(3,nx)) , 1)
      rg = np.append(rg,rl,axis=0)

    resid = rg.flatten(order='F') 
    #resid,_,_=  computeVelocityLocal(u) 
    #resid = np.reshape(resid,(3,nx))
    #resid[:,0] *= lam
    #resid[:,-1] *= lam
    if l1:
      return np.sqrt(np.abs(resid.flatten()))
    else:
      return resid.flatten()  

  def residualGalerkin(xhat):
    resid = residual(xhat,False)
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
data = np.load('../snapshots_vfine.npz')
u_fom = data['snapshots']
u_fom = np.reshape(u_fom,(nx,3,np.shape(u_fom)[-1]),order='F')
#params = np.array([0.5,0.875,1.25,1.625])
params = np.linspace(0.5,1.625,25)
#params = np.linspace(0.2,2.0,10)
snapshots_resmin = np.zeros((3*N,np.size(params)))
snapshots_galerkin = np.zeros((3*N,np.size(params)))
snapshots_ml = np.zeros((3*N,np.size(params)))
residual_resmin = np.zeros(np.size(params))
residual_galerkin = np.zeros(np.size(params))
residual_ml = np.zeros(np.size(params))

u = np.zeros((3,N))

  
xi,mui = np.meshgrid(x/10.,params)
xi = xi.flatten()
mui = mui.flatten() 
input_features = np.append(xi[:,None],mui[:,None],axis=1)
Phi1 = model.createBasis(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
Phi = np.zeros((3*np.size(xi),nbasis))
Phi = np.reshape(Phi1,(3*np.size(xi),nbasis),order='F')
Phi,s,_ = np.linalg.svd(Phi,full_matrices=False)
u_ml = model.forward(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
u_ml = np.reshape(u_ml,(3*np.size(xi)),order='F')
xhat0 = np.dot(Phi.transpose(),u_ml)
umlp = np.reshape( np.dot(Phi,xhat0) , (np.size(xi),3),order='F')
u_resmin = solveSystem(params,Phi,'resmin')
u_resmin = np.rollaxis( np.reshape(u_resmin,(np.size(x),np.size(params),3),order='F') , 2,1)

u_galerkin = solveSystem(params,Phi,'Galerkin')
u_galerkin = np.rollaxis( np.reshape(u_galerkin,(np.size(x),np.size(params),3),order='F') , 2,1)

u_ml = np.rollaxis( np.reshape(u_ml,(np.size(x),np.size(params),3),order='F') , 2,1)


print('Res Min Error  = ' + str(np.linalg.norm(u_resmin - u_fom)))
print('Galerkin Error = ' + str(np.linalg.norm(u_galerkin - u_fom)))
print('ML Error       = ' + str(np.linalg.norm(u_ml - u_fom)))

print('H1 Res Min Error  = ' + str(np.linalg.norm(diff(u_resmin[0:1024] - u_fom[0:1024]))))
print('H1 Galerkin Error = ' + str(np.linalg.norm(diff(u_galerkin[0:1024] - u_fom[0:1024]))))
print('H1 ML Error       = ' + str(np.linalg.norm(diff(u_ml[0:1024] - u_fom[0:1024]))))

#print('Res Min Residual  = ' + str(np.linalg.norm(residual_resmin)))
#print('Galerkin Residual = ' + str(np.linalg.norm(residual_galerkin)))
#print('ML Residual       = ' + str(np.linalg.norm(residual_ml)))
#'''
