from pylab import *
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


#global parameters
gamma = 1.4

N = 1024
x = np.linspace(0,10.,N)
dx = x[1]

def solveSystem(mu):
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
 
  t = 0
  dt = 5.e-3
  et = 100
  rk4const = np.array([1./4.,1./3.,1./2.,1.])

  counter = 0
  # do pesudo time stepping to get a good guess 
  fnorm = 10
  while fnorm >= 1e-8:
    u0 = u.flatten()*1.
    for i in range(0,4):
      f = computeVelocityLocal(u)
      u = u0 + dt*rk4const[i]*f
    t += dt
    fnorm = np.linalg.norm(f)
    if (counter%500 == 0):
      print('Residual norm = ' + str(fnorm))
    counter += 1
  print('Final residual norm = ' + str(fnorm))
  # run through a final newton krylov step to get converged solution
  #u = scipy.optimize.newton_krylov(computeVelocityLocal,u.flatten(),verbose=4)
  return u

#params = np.array([0.5,0.875,1.25,1.625])
params = np.linspace(0.5,1.625,25)
snapshots = np.zeros((3*N,np.size(params)))
u = np.zeros((3,N))
for i in range(np.size(params)):
  print('On parameter ' + str(i) + ' of ' + str(np.size(params)))
  snapshots[:,i] = solveSystem(params[i])
    

np.savez('snapshots_vfine',snapshots=snapshots,params=params,x=x)
