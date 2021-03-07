from pylab import *
import scipy.optimize
def uExact(t):
  u = np.zeros(nx)
  for i in range(0,nx):
    if (x[i] - t >= 0.05 and x[i] -  t < 0.35):
      u[i] = 1.
  return u 



def residual(u):
  return u - un - 0.5*dt*( f(u) + f(un) )

def f(u):
  ux = np.zeros(np.size(u))
  ux[1::] = 0.5/dx*(u[1::]**2 - u[0:-1]**2)
  ux[0] = 0.5/dx*(u[0]**2 - mu1**2)
  return -ux + 0.02*exp(mu2*x)


nx = 256
x = linspace(0,100,nx)
dx = x[1]

uIC = np.ones(nx)

et = 35 
rk4const = np.array([1./4.,1./3.,1./2.,1.])
#usave = np.zeros((nx,0,0))

#mu1 = 4.3
#mu2 = 0.021
#mu1 = 5.
#mu2 = 0.0225
mu1 = 6.
mu2 = 0.04##25

mu1a = np.array([mu1])
mu2a = np.array([mu2])
save_freq = 1
for k in range(0,np.size(mu1a)):
  print('On run no ' + str(k))
  t = 0
  u = uIC*1.
  dt = 0.07
  mu1 = mu1a[k]
  mu2 = mu2a[k]
  usavel = np.zeros((nx,0))
  counter =0
  tsave = np.zeros(0)
  while (t <= et):
    un = u*1.
    u = scipy.optimize.newton_krylov(residual,un,verbose=4)
    #for i in range(0,4):
    #  u = u0 + dt*rk4const[i]*f(u)
    t += dt
    if (counter  % save_freq == 0):
      usavel = np.append(usavel,u[:,None],axis=1)
      tsave = np.append(tsave,t)
    counter += 1

  if k == 0:
    usave = usavel[:,:,None]*1.
  else:
    usave = np.append(usave,usavel[:,:,None],axis=2)

np.savez('snapshots_st_cn_test',snapshots=usave,mu1a=mu1a,mu2a=mu2a,x=x,t=tsave)

'''
U,S,V = np.linalg.svd(usave)

rel_e = np.cumsum(S**2)/np.sum(S**2)
tol = 0.9999999
K = np.size(rel_e[rel_e < tol ])
Phi = U[:,0:K]
xhat = np.dot(Phi.transpose(),usave)
utilde = np.dot(Phi,xhat)
np.savez('snapshots_st',snapshots=usave,xhat=xhat)
'''
