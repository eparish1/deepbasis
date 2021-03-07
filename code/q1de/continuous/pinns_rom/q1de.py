from pylab import *
import numpy as np
from scipy.interpolate import lagrange
import scipy.optimize
from scipy.sparse.linalg import LinearOperator
try: 
  from adolc import * 
except:
#  if (MPI.COMM_WORLD.Get_rank() == 0):
    print("adolc not found, can't use adolc automatic differentiation")

def rusanovFlux(gamma,UL,UR,args=None):
# PURPOSE: This function calculates the flux for the Euler equations
# using the Rusanov flux function
#
# INPUTS:
#    UL: conservative state vector in left cell
#    UR: conservative state vector in right cell
#    n: normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbance
#
  gmi = gamma-1.0
  #process left state
  rL = UL[0] + 1e-30
  uL = UL[1]/rL
  unL = uL

  qL = (UL[1]*UL[1])**0.5/rL
  pL = (gamma-1)*(UL[2] - 0.5*rL*qL**2.)
  rHL = UL[2] + pL
  HL = rHL/rL
  cL =(gamma*pL/rL)**0.5
  # left flux
  FL = np.zeros(np.shape(UL),dtype=UL.dtype)
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL
  FL[2] = rHL*unL

  # process right state
  rR = UR[0] + 1e-30
  uR = UR[1]/rR
  unR = uR
  qR = (UR[1]*UR[1])**0.5/rR
  pR = (gamma-1)*(UR[2] - 0.5*rR*qR**2.)
  rHR = UR[2] + pR
  HR = rHR/rR
  cR = (gamma*pR/rR)**0.5
  # right flux
  FR = np.zeros(np.shape(UR),dtype=UR.dtype)
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR
  FR[2] = rHR*unR

  # difference in states
  du = UR - UL

  # Roe average
  di     = (rR/rL)**0.5
  d1     = 1.0/(1.0+di)

  ui     = (di*uR + uL)*d1
  Hi     = (di*HR + HL)*d1

  af     = 0.5*(ui*ui)
  ucp    = ui
  c2     = gmi*(Hi - af)
  ci     = (c2)**0.5
  ci1    = 1.0/(ci + 1.e-30)

  #% eigenvalues

  sh = np.shape(ucp)
  lsh = np.append(3,sh)
  smax = np.abs(ucp) + np.abs(ci)
  F = np.zeros(np.shape(UL),dtype=UL.dtype)
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
  return F

def computeVelocity(U,dlnAdx,dx):
  N = np.size(dlnAdx)
  U = np.reshape(U,(3,N))
  gamma = 1.4
  f = np.zeros(np.shape(U),dtype=U.dtype)
  UL = np.zeros(3,dtype=U.dtype)
  UR = np.zeros(3,dtype=U.dtype)

  ## boundary conditions
  UL[0] = 1.
  UL[1] = U[1,0]
  pL = 1.
  UL[2] = pL/(gamma - 1.) + 0.5*UL[0]*UL[1]**2

  pR = 0.7
  UR[0] = U[0,-1]
  UR[1] = U[1,-1]
  UR[2] = pR/(gamma - 1.) + 0.5*UR[0]*UR[1]**2
  U = np.append(UL[:,None],U,axis=1)
  U = np.append(U,UR[:,None],axis=1)

  ## compute flux and forcing
  Flux = rusanovFlux(gamma,U[:,0:-1],U[:,1::])
  f[:] = -1./dx*(Flux[:,1::]- Flux[:,0:-1])

  ## add forcing term
  U = U[:,1:-1]
  q = (U[1]*U[1])**0.5/U[0]
  p = (gamma-1)*(U[2] - 0.5*U[0]*q**2.)
  H = (U[2] + p)/U[0]
  G = np.zeros(np.shape(U),dtype=U.dtype)
  G[0] = U[1]
  G[1] = U[1]**2/U[0]
  G[2] = U[1]*H
  G *= dlnAdx[None,:] 
  f[:] -= G
  return f.flatten()#*1./diag(J)

'''
class preconditionedVelocity:
  def __init__(self,J,U):
    self.J = J
    self.counter = 0
    self.U = U

    def updateState(u,r):
      self.U = u
      if (self.counter % 50 == 0):
        print('Recomputing jacobian',self.counter)
        self.J = jacobian(1,self.U.flatten())
      self.counter += 1 

    def mv(v):
      #return np.dot(diag(self.J)*eye(3*N),v)#1./diag(self.J)*v
      return diag(self.J)*v

    def velocity(U):
      f = computeVelocity(U)
      return f

    self.velocity = velocity
    self.updateState = updateState
    self.PC = LinearOperator((3*N,3*N),matvec=mv)


ax = adouble(u.flatten())
trace_on(1)
independent(ax)
ay = computeVelocity(ax)
dependent(ay)
trace_off()
J = jacobian(1,u.flatten())
'''

