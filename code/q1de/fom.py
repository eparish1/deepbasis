from pylab import *

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


nx = 512
x = linspace(0,1,nx)
dx = x[1]

uIC = np.zeros(nx)
for i in range(0,nx):
  if (x[i] < 0.35):
    uIC[i] = 1.

t = 0
et = 0.5
rk4const = np.array([1./4.,1./3.,1./2.,1.])
usave = np.zeros((nx,0))
u2save = np.zeros((nx,0))

u = uIC*1.
#CFL = c*dt/dx
CFL = 0.5
dt = CFL*dx
while (t <= et):
  u0 = u*1.
  for i in range(0,4):
    u = u0 + dt*rk4const[i]*f(u)
  usave = np.append(usave,u[:,None],axis=1)
  u2 = uExact(t)
  u2save = np.append(u2save,u2[:,None],axis=1)

  t += dt

U,S,V = np.linalg.svd(usave)
nt = np.shape(usave)[1]
np.savez('snapshots',snapshots=usave[:,0:int(nt/2)],snapshots_test=usave[:,int(nt/2)::])
