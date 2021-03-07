from pylab import *
import scipy.optimize
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deep_nn import DeepNN
import time
close("all")
torch.set_default_dtype(torch.float32)
def uExact(t):
  u = np.zeros(nx)
  for i in range(0,nx):
    if (x[i] - t >= 0.05 and x[i] -  t < 0.35):
      u[i] = 1.
  return u 


def f(u):
  ux = np.zeros(np.size(u))
  ux[1::] = 1./dx*(u[1::] - u[0:-1])
  ux[0] = 1./dx*(u[0] - 1.)
  return -ux

def buildA(nx,dx):
  A = np.zeros((nx,nx))
  for i in range(0,nx):
    A[i,i] = 1./dx
    if (i > 0):
      A[i,i-1] = -1./dx
  return -A

def createMatrixST(nx,nt):
  A  = buildA(nx,dx)
  rowA,colA = np.nonzero(A)
  Adata=A[rowA,colA] 
  #AST = np.zeros((nx*nt,nx*nt))
  rowG = np.zeros(0,dtype='int')
  colG = np.zeros(0,dtype='int')
  dataG = np.zeros(0)
  for i in range(0,nt):
    ## Add dt term
    row_start_index = i*nx
    rowl = np.array(range(row_start_index,row_start_index + nx)) 
    col_start_index = i*nx
    coll = np.array(range(col_start_index,col_start_index + nx)) 
    datal = np.ones(np.size(rowl))
    rowG = np.append(rowG,rowl)
    colG = np.append(colG,coll)
    dataG = np.append(dataG,datal)
    ## add A matrix
    rowl = row_start_index + rowA
    coll = col_start_index + colA
    datal = -0.5*dt*Adata
    rowG = np.append(rowG,rowl)
    colG = np.append(colG,coll)
    dataG = np.append(dataG,datal)
    if (i > 0):
      row_start_index = i*nx
      rowl = np.array(range(row_start_index,row_start_index + nx)) 
      col_start_index = i*nx - nx
      coll = np.array(range(col_start_index,col_start_index + nx)) 
      datal = -np.ones(np.size(rowl))*1
      rowG = np.append(rowG,rowl)
      colG = np.append(colG,coll)
      dataG = np.append(dataG,datal)
      rowl = row_start_index + rowA
      coll = col_start_index + colA
      datal = -0.5*dt*Adata
      rowG = np.append(rowG,rowl)
      colG = np.append(colG,coll)
      dataG = np.append(dataG,datal)

  AST = scipy.sparse.csr_matrix((dataG,(rowG,colG)),(nx*nt,nx*nt))
  return AST   


def spaceTimeResidual(u):
  u = np.reshape(u,(nt,nx))
  residual = np.zeros(0)
  u0 = uIC*1.
  for i in range(0,nt):
    local_residual = u[i] - u0 - 0.5*dt*(f(u[i]) + f(u0))
    residual = np.append(residual,local_residual)
    u0 = u[i]*1.
  return residual

nx = 512
x = linspace(0,1,nx)
dx = x[1]
CFL = 0.5
dt = CFL*dx

uIC = np.zeros(nx)
for i in range(0,nx):
  if (x[i] < 0.35):
    uIC[i] = 1.

#uIC += 0.05*np.cos(2.*np.pi*x*10)

et = 0.5
nt = int(ceil(et/dt))
t = np.linspace(0,et,nt)
dt = t[1]
x,t = np.meshgrid(x,t)#,indexing='ij')
x = x.flatten()
t = t.flatten()
input_features = np.append(x[:,None],t[:,None],axis=1)

uguess = np.zeros((nt,nx))
uguess[:] = uIC[None]

AST = createMatrixST(nx,nt)
forcing =  AST.dot(uguess.flatten()) - spaceTimeResidual(uguess.flatten()) 
sol = scipy.sparse.linalg.spsolve(AST,forcing)
ufom = np.reshape(sol,(nt,nx))



#usol = scipy.optimize.newton_krylov(spaceTimeResidual,uguess.flatten(),verbose=4) 

#===
depth=4
nbasis=4
model = DeepNN(depth,4)
np.random.seed(1)
try:
  model.load_state_dict(torch.load('tmp_model_depth' + str(depth),map_location='cpu'))
  print('Succesfully loaded model!')
except:
  print('Failed to load model')

device = 'cpu'
model.to(device)
Phi = model.createBasis(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
uml = model.forward(torch.tensor(input_features,dtype=torch.float32)).detach().numpy()
uml = np.reshape(uml,(nt,nx))
Phi,_,_ = np.linalg.svd(Phi,full_matrices=False)
ASTROM = np.dot(Phi.transpose(),AST.dot(Phi))
forcing_rom = np.dot(Phi.transpose(),forcing)
urom = np.linalg.solve(ASTROM,forcing_rom)
urom = np.reshape( np.dot(Phi,urom) , (nt,nx))


ASTROM_LS = np.dot(Phi.transpose(),AST.transpose().dot(AST.dot(Phi)))
forcing_rom_ls = np.dot(Phi.transpose(),AST.transpose().dot(forcing))
urom_ls = np.linalg.solve(ASTROM_LS,forcing_rom_ls)
urom_ls= np.reshape( np.dot(Phi,urom_ls) , (nt,nx))


#t = 0
#et = 0.5
#rk4const = np.array([1./4.,1./3.,1./2.,1.])
#usave = np.zeros((nx,0))
#u2save = np.zeros((nx,0))
#
#u = uIC*1.
##CFL = c*dt/dx
#CFL = 0.5
#dt = CFL*dx
#tsave = np.zeros(0)
#while (t <= et):
#  u0 = u*1.
#  for i in range(0,4):
#    u = u0 + dt*rk4const[i]*f(u)
#  usave = np.append(usave,u[:,None],axis=1)
#  u2 = uExact(t)
#  u2save = np.append(u2save,u2[:,None],axis=1)
#  t += dt
#  tsave = np.append(tsave,t)
#
#U,S,V = np.linalg.svd(usave)
#nt = np.shape(usave)[1]
#np.savez('snapshots',snapshots=usave[:,:],t=tsave,x=x)
