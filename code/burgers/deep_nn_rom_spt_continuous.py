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
torch.set_default_dtype(torch.float64)
axis_font = {'size':'24'}
def f(u,mu1,mu2):
  if (np.size(mu1) == 1):
    ux = np.zeros(np.size(u))
    ux[1::] = 0.5/dx*(u[1::]**2 - u[0:-1]**2)
    ux[0] = 0.5/dx*(u[0]**2 - mu1**2)
    return -ux + 0.02*exp(mu2*x)
  else:
    ux = np.zeros((np.shape(u)))
    ux[1::] = 0.5/dx*(u[1::]**2 - u[0:-1]**2)
    ux[0] = 0.5/dx*(u[0]**2 - mu1[None]**2)

    #uxx = np.zeros((np.shape(u)))
    #uxx[1:-1] = 0.001/dx**2*(u[2::] - 2.*u[1:-1] + u[0:-2])
    return -ux*0. + 0.02*exp(mu2[None,None]*x[:,None,None,None]) 
   


def spaceTimeResidual(u):
  u = np.reshape(u,(nx,nt,np1,np2),order='F')
  residual = np.zeros((nx,nt,np1,np2))
  mu2v,mu1v = np.meshgrid(mu2a,mu1a)
  u0 = uIC[:,None,None]*np.ones((nx,np1,np2))
  for i in range(0,nt):
    local_residual = u[:,i,:,:] - u0 - 0.5*dt*(f(u[:,i,:,:],mu1v,mu2v) + f(u0,mu1v,mu2v))
    residual[:,i,:,:] = local_residual
    u0 = u[:,i,:,:]*1.
  '''
  for k in range(0,np2):
    for l in range(0,np1):
      u0 = uIC*1.
      for i in range(0,nt):
        local_residual = u[:,i,l,k] - u0 - 0.5*dt*(f(u[:,i,l,k],mu1a[l],mu2a[k]) + f(u0,mu1a[l],mu2a[k]))
        residual[:,i,l,k] = local_residual
        u0 = u[:,i,l,k]*1.
  '''
  return residual.flatten(order='F')

def residual(u):
  return u - un - 0.5*dt*( f(u,mu1,mu2) + f(un,mu1,mu2) )


def spaceTimeResidualLsRom(xhat):
  uml = np.dot(Phi,xhat)
  udotml = np.dot(PhiDot,xhat)

  uml = np.reshape(uml,(nx,nt,np1,np2),order='F')
  residual = spaceTimeResidual(uml)
  return residual.flatten(order='F')

def spaceTimeResidualL1Rom(xhat):
  uml = np.dot(Phi,xhat)
  uml = np.reshape(uml,(nx,nt,np1,np2),order='F')
  residual = spaceTimeResidual(uml)
  return np.sqrt(np.abs(residual.flatten(order='F')))


def spaceTimeResidualGalerkinRom(xhat):
  uml = np.dot(Phi,xhat)
  mu2v,mu1v = np.meshgrid(mu2a,mu1a)
  uml = np.reshape(uml,(nx,nt,np1,np2),order='F')

  velocity = f(uml,mu1v,mu2v).flatten(order='F')
  uet = np.dot(Phiet,xhat)
  uICl =  (uIC[:,None,None]*np.ones((nx,np1,np2))).flatten(order='F')
  #residual = dzeta_dt*np.dot(Phi.transpose(),np.dot(PhiDot,xhat)) - np.dot(Phi.transpose(),velocity)
  grad_x = np.einsum('i,jil->jl',xhat,PhiTPC)
  grad_x = np.einsum('l,jl->j',xhat,grad_x)
  #print(np.amax( np.dot(Phi.transpose(),velocity)),np.amax(grad_x))
  #pause(0.001)
  #clf()
  u0  = np.dot(Phi0,xhat)
  residual = -dzeta_dt*np.dot(PhiDot.transpose(),np.dot(Phi,xhat))*dt*dx - np.dot(Phi.transpose(),velocity)*dt*dx
  residual += -dzeta_dx*np.dot(Phi_grad_x.transpose(),0.5*np.dot(Phi,xhat)**2)*dx*dt 

  residual += np.dot(Phiet.transpose(),uet)*dx
  residual -= np.dot(Phi0.transpose(),uICl)*dx

  # left and right boundaries
  uL = np.dot(PhixL,xhat)
  residual += np.dot(PhixR.transpose(),0.5*np.dot(PhixR,xhat)**2)*dt
  residual -= np.dot(PhixL.transpose(),0.5*(input_features_xL[:,2]*(5.5 - 4.25) + 4.25)**2)*dt

#  residual -= np.dot(PhixL.transpose(),0.5*( 0.5*(input_features_xL[:,2]*(5.5 - 4.25) + 4.25 + uL))**2 )*dt
#  residual -= np.dot(PhixL.transpose(),0.5*( np.dot(PhixL,xhat)**2) )*dt

  return np.dot(massInv,residual) 

def spaceTimeResidualDiscreteGalerkinRom(xhat):
  uml = np.dot(Phi,xhat)
  residual = spaceTimeResidual(uml)
  return np.dot(Phi.transpose(),residual)  


modelNo = 1
extrapolate = True#True
normalize = True#True
box = False
if box:
  model_type = 'box'
else:
  model_type = 'elu'
depth=5
nbasis=10
device = 'cpu'
if modelNo == "Ensemble":
  models = [ None ] *11
  for i in range(0,1):
    models[i] = DeepNN(depth,nbasis,box)
    models[i].load_state_dict(torch.load('synapse_models/' + model_type + '/depth_6_nbasis_8_no_' + str(i) + '_trained',map_location='cpu'))
    models[i].to(device)
  for i in range(1,11):
    models[i] = DeepNN(4,5,box)
    models[i].load_state_dict(torch.load('synapse_models/' + model_type + '_short2/depth_4_nbasis_5_no_' + str(i-1) + '_trained',map_location='cpu'))
    models[i].to(device)
#  models = [ None ] *5
#  for i in range(0,5):
#    models[i] = DeepNN(depth,nbasis,box)
#    models[i].load_state_dict(torch.load('synapse_models/' + model_type + '/depth_6_nbasis_8_no_' + str(i) + '_trained',map_location='cpu'))
#    models[i].to(device)
else:
  model = DeepNN(depth,nbasis,box)
  try:
  #  model.load_state_dict(torch.load('tmp_model_depth' + str(depth),map_location='cpu'))
  #  model.load_state_dict(torch.load('tmp_model_depth6_nbasis_8',map_location='cpu'))
  #  model.load_state_dict(torch.load('depth_6_nbasis_8_no_1_trained',map_location='cpu'))
    #model.load_state_dict(torch.load('synapse_models/' + model_type + '/depth_6_nbasis_8_no_' + str(modelNo) + '_trained',map_location='cpu'))
    model.load_state_dict(torch.load('synapse_models/basis_study/depth_' + str(depth) + '_nbasis_' + str(nbasis) + '_no_' + str(modelNo) + '_trained',map_location='cpu'))

  #  model.load_state_dict(torch.load('tmpdepth_4_nbasis_10_no_0',map_location='cpu'))
    model.to(device)
    print('Succesfully loaded model!')
  except:
    print('Failed to load model')

nx = 256
x = linspace(0,100,nx)
dx = x[1]
uIC = np.ones(nx)
if extrapolate:
  et = 35. 
  mu1a = np.linspace(4,6.,6)
  mu2a = np.linspace(0.01,0.04,6)
  #mu1a = np.linspace(4.25,5.5,2) 
  #mu2a = np.linspace(0.015,0.015 + 0.015,2)

else:
  et = 35./2.
  mu1a = np.linspace(4.25,5.5,6) 
  mu2a = np.linspace(0.015,0.015 + 0.015,6)

np1 = np.size(mu1a)
np2 = np.size(mu2a)
save_freq = 1
dx = x[1]
t = 0
u = uIC*1.
dt = 0.07
usavel = np.zeros((nx,0))
counter =0
tsave = np.zeros(0)

#===
data = np.load('snapshots_st_cn.npz')
#uIC = data['snapshots'][:,0,0]
if extrapolate:
  ta = np.linspace(0,35-dt,500)
else:
  ta = np.linspace(0,17.5-dt,250)
#ta = data['t']#[0:250]
nt = np.size(ta)

### Construct grids
mu1vtest,mu2vtest,tvtest,xvtest = np.meshgrid(( mu1a - 4.25)/(5.5 - 4.25),( mu2a - 0.015)/(0.03-0.015),ta/35.,x/100.)
xvtest = xvtest.flatten()
tvtest = tvtest.flatten()
mu1vtest = mu1vtest.flatten()
mu2vtest = mu2vtest.flatten()

input_features = np.append(xvtest[:,None],tvtest[:,None],axis=1)
input_features = np.append(input_features,mu1vtest[:,None],axis=1)
input_features = np.append(input_features,mu2vtest[:,None],axis=1)

mu1vtest_t0,mu2vtest_t0,tvtest_t0,xvtest_t0 = np.meshgrid(( mu1a - 4.25)/(5.5 - 4.25),( mu2a - 0.015)/(0.03-0.015),np.linspace(0,0,1)/35.,x/100.)
xvtest_t0 = xvtest_t0.flatten()
tvtest_t0 = tvtest_t0.flatten()
mu1vtest_t0 = mu1vtest_t0.flatten()
mu2vtest_t0 = mu2vtest_t0.flatten()
input_features_t0 = np.append(xvtest_t0[:,None],tvtest_t0[:,None],axis=1)
input_features_t0 = np.append(input_features_t0,mu1vtest_t0[:,None],axis=1)
input_features_t0 = np.append(input_features_t0,mu2vtest_t0[:,None],axis=1)


mu1vtest_et,mu2vtest_et,tvtest_et,xvtest_et = np.meshgrid(( mu1a - 4.25)/(5.5 - 4.25),( mu2a - 0.015)/(0.03-0.015),np.linspace(et-dt,et-dt,1)/35.,x/100.)
xvtest_et = xvtest_et.flatten()
tvtest_et = tvtest_et.flatten()
mu1vtest_et = mu1vtest_et.flatten()
mu2vtest_et = mu2vtest_et.flatten()

input_features_et = np.append(xvtest_et[:,None],tvtest_et[:,None],axis=1)
input_features_et = np.append(input_features_et,mu1vtest_et[:,None],axis=1)
input_features_et = np.append(input_features_et,mu2vtest_et[:,None],axis=1)

mu1vtest_xR,mu2vtest_xR,tvtest_xR,xvtest_xR = np.meshgrid(( mu1a - 4.25)/(5.5 - 4.25),( mu2a - 0.015)/(0.03-0.015),ta/35.,x[-1::]/100.)
xvtest_xR = xvtest_xR.flatten()
tvtest_xR = tvtest_xR.flatten()
mu1vtest_xR = mu1vtest_xR.flatten()
mu2vtest_xR = mu2vtest_xR.flatten()

input_features_xR = np.append(xvtest_xR[:,None],tvtest_xR[:,None],axis=1)
input_features_xR = np.append(input_features_xR,mu1vtest_xR[:,None],axis=1)
input_features_xR = np.append(input_features_xR,mu2vtest_xR[:,None],axis=1)

mu1vtest_xL,mu2vtest_xL,tvtest_xL,xvtest_xL = np.meshgrid(( mu1a - 4.25)/(5.5 - 4.25),( mu2a - 0.015)/(0.03-0.015),ta/35.,x[0:1]/100.)
xvtest_xL = xvtest_xL.flatten()
tvtest_xL = tvtest_xL.flatten()
mu1vtest_xL = mu1vtest_xL.flatten()
mu2vtest_xL = mu2vtest_xL.flatten()

input_features_xL = np.append(xvtest_xL[:,None],tvtest_xL[:,None],axis=1)
input_features_xL = np.append(input_features_xL,mu1vtest_xL[:,None],axis=1)
input_features_xL = np.append(input_features_xL,mu2vtest_xL[:,None],axis=1)


Phi = model.createBasis(torch.tensor(input_features,dtype=torch.float64)).detach().numpy()
Phi0 = model.createBasis(torch.tensor(input_features_t0,dtype=torch.float64)).detach().numpy()
Phiet = model.createBasis(torch.tensor(input_features_et,dtype=torch.float64)).detach().numpy()
PhixR = model.createBasis(torch.tensor(input_features_xR,dtype=torch.float64)).detach().numpy()
PhixL = model.createBasis(torch.tensor(input_features_xL,dtype=torch.float64)).detach().numpy()

PhiDot = np.zeros(np.shape(Phi))
Phi_grad_x = np.zeros(np.shape(Phi))

for i in range(0,np.shape(Phi)[1]):
  print('Computing grad ' + str(i))
  if2t = torch.tensor(input_features,dtype=torch.float64,requires_grad=True)
  Phitmp = model.createBasis(if2t)
  Phitmp[:,i].backward(gradient=torch.ones_like(Phitmp[:,i]),retain_graph=True)
  PhiDot[:,i] = if2t.grad[:,1].detach().numpy()
  Phi_grad_x[:,i] = if2t.grad[:,0].detach().numpy()

  if2t.grad.data.zero_()

dzeta_dt = 1./35.
dzeta_dx = 1./100.

PhiDot = PhiDot

PhiT = np.zeros( (np.shape(Phi)[0],np.shape(Phi)[1],np.shape(Phi)[1]))
for i in range(0,np.shape(Phi)[1]):
  for j in range(0,np.shape(Phi)[1]):
    PhiT[:,i,j] = (Phi[:,i])*Phi[:,j]

PhiTPC = np.einsum('ij,ikl->jkl',Phi_grad_x,PhiT)

#orthogonal l2 projection
uml = model.forward(torch.tensor(input_features,dtype=torch.float64)).detach().numpy()[:,0]

mass = np.dot(Phi.transpose(),Phi)
massInv = np.linalg.inv(mass)
rhs = np.dot(Phi.transpose(),uml).flatten()
xhat0 = np.linalg.solve(mass,rhs)

u0 = ( uIC[:,None,None]*np.ones((nx,np1,np2)) ).flatten(order='F')

t1 = dzeta_dt*np.dot(Phi.transpose(),np.dot(PhiDot,xhat0))*dt*dx
t1 += dzeta_dx*np.dot(Phi.transpose(),np.dot(Phi,xhat0)*np.dot(Phi_grad_x,xhat0))*dx*dt

t2 =  -dzeta_dt*np.dot(PhiDot.transpose(),np.dot(Phi,xhat0))*dt*dx
t2 += -dzeta_dx*np.dot(Phi_grad_x.transpose(),0.5*np.dot(Phi,xhat0)**2)*dx*dt

t2 += np.dot(Phiet.transpose(),np.dot(Phiet,xhat0))*dx
t2 -= np.dot(Phi0.transpose(),np.dot(Phi0,xhat0))*dx

t2 += np.dot(PhixR.transpose(),0.5*np.dot(PhixR,xhat0)**2)*dt
t2 -= np.dot(PhixL.transpose(),0.5*np.dot(PhixL,xhat0)**2 )*dt






t0 = time.time()
#xhat = scipy.optimize.least_squares(spaceTimeResidualGalerkinRom,xhat0.flatten(),verbose=2).x

#xhat = scipy.optimize.broyden1(spaceTimeResidualGalerkinRom,xhat0.flatten(),verbose=1)
xhat = scipy.optimize.newton_krylov(spaceTimeResidualGalerkinRom,xhat0.flatten(),verbose=1,inner_maxiter=100,maxiter=2500)
urom = np.dot(Phi,xhat)
urom = np.reshape(urom,(nx,nt,np1,np2),order='F')


grad_x = np.einsum('i,jil->jl',xhat0,PhiT)
grad_x = np.einsum('l,jl->j',xhat0,grad_x)
uux = np.reshape(grad_x,(nx,nt,np1,np2),order='F')
uml = np.reshape(uml,(nx,nt,np1,np2),order='F')

#  urom = 0.
nt = int(ceil(et/dt))
ufom = np.zeros((nx,nt,np1,np2))
un = uIC*1.
for k in range(0,np2):
  for l in range(0,np1):
    t = 0
    u = uIC*1.
    dt = 0.07
    counter =0
    tsave = np.zeros(0)
    mu1 = mu1a[l]
    mu2 = mu2a[k]
    while (t <= et):
      un = u*1.
      u = scipy.optimize.newton_krylov(residual,un,verbose=0)
      t += dt
      ufom[:,counter,l,k] = u[:]
      tsave = np.append(tsave,t)
      counter += 1
