from pylab import *
import sys
sys.path.append('../../')
from matplotlib import rc
from deep_nn import DeepNN
import torch
torch.set_default_dtype(torch.float32)
# rc('text', usetex=True)
close("all")
axis_font = {'size':'20','family':'serif'}
rerun = True
depthA = np.array([1,2,3,4])
basisA = np.array([5,10,15, 20, 25])
depthAG,basisAG = np.meshgrid(depthA,basisA,indexing='ij')
ndepth = np.size(depthA)
nbasis = np.size(basisA)

#rc('text.latex', preamble='\usepackage{amsmath},\usepackage{amssymb}')
#####
nx = 256
x = linspace(0,100,nx)
dx = x[1]
uIC = np.ones(nx)
et = 35.
mu1a = np.linspace(4,6.,8)
mu2a = np.linspace(0.01,0.04,8)
np1 = np.size(mu1a)
np2 = np.size(mu2a)
save_freq = 1
dx = x[1]
t = 0
u = uIC*1.
dt = 0.07
ta = np.linspace(dt,35,500)
nt = np.size(ta)
normalize=True
if normalize == True:
  mu1vtest,mu2vtest,tvtest,xvtest = np.meshgrid(( mu1a - 4.25)/(5.5 - 4.25),( mu2a - 0.015)/(0.03-0.015),ta/35.,x/100.)
else:
  mu1vtest,mu2vtest,tvtest,xvtest = np.meshgrid( mu1a ,mu2a,ta,x)

xvtest = xvtest.flatten()
tvtest = tvtest.flatten()
mu1vtest = mu1vtest.flatten()
mu2vtest = mu2vtest.flatten()

input_features = np.append(xvtest[:,None],tvtest[:,None],axis=1)
input_features = np.append(input_features,mu1vtest[:,None],axis=1)
input_features = np.append(input_features,mu2vtest[:,None],axis=1)

input_features_torch = torch.tensor(input_features,dtype=torch.float32)
nmodels = 1
MSE_a = np.zeros((ndepth,nbasis,nmodels))
MSE_Pa = np.zeros((ndepth,nbasis,nmodels))
MSE_MLa = np.zeros((ndepth,nbasis,nmodels))

ufom =  np.reshape( np.load('snapshots_st_cn_ood.npz')['snapshots'], (nx,500,8,8)).flatten(order='F')

models_stats = [None]*np.size(depthA)
models = [None]*np.size(depthA)
for i in range(0,np.size(models_stats)):
  models_stats[i] = [None]*np.size(basisA)
  models[i] = [None]*np.size(basisA)
  for j in range(0,np.size(basisA)):
    models[i][j] = [None]*nmodels
    models_stats[i][j] = [None]*nmodels

print(len(models_stats))
print(len(models_stats[0]))
print(len(models_stats[0][0]))
print(models_stats)
keyboard

for i in range(0,ndepth):
  for j in range(0,nbasis):
    for k in range(0,nmodels):
      print(i,j,k)
      models_stats[i][j][k] = np.load('stats_depth_' + str(depthA[i]) + '_nbasis_' + str(basisA[j]) + '_no_' + str(k) + '.npz')
      models[i][j][k] = DeepNN(depthA[i],basisA[j],False)
      models[i][j][k].load_state_dict(torch.load('depth_' + str(depthA[i]) + '_nbasis_' + str(basisA[j]) + '_no_' + str(k) + '_trained',map_location='cpu'))
      models[i][j][k].to('cpu')
      Phi = models[i][j][k].createBasis(torch.tensor(input_features_torch,dtype=torch.float32)).detach().numpy()
      Phi,_,_ = np.linalg.svd(Phi,full_matrices=False)
      uml = models[i][j][k].forward(input_features_torch).detach().numpy()
      ufom_p = np.dot(Phi,np.dot(Phi.transpose(),ufom))
      MSE_Pa[i,j,k] = np.mean( (ufom_p - ufom )**2 )
      MSE_MLa[i,j,k] = np.mean( (uml[:,0] - ufom )**2 )
      MSE_a[i,j,k] = models_stats[i][j][k]['train_loss'][-1]
      print(MSE_a[i,j,k])
MSE_Pa_mean = np.mean(MSE_Pa,axis=2)
MSE_MLa_mean = np.mean(MSE_MLa,axis=2)
MSE_a_mean = np.mean(MSE_a,axis=2)
print(MSE_a_mean)

ca = ['red','blue','green','purple','black','orange']
nsamples_training = 128000./ 10000
for i in range(0,ndepth):
  plot(basisA,MSE_Pa_mean[i],color=ca[i],label='Depth = ' + str(depthA[i]) )
  plot(basisA,MSE_a_mean[i],color=ca[i],ls='--',label='(ML) Depth = ' + str(depthA[i]) )
#  plot(basisA,MSE_MLa[i],color=ca[i],ls='--',label='(ML) Depth = ' + str(depthA[i]) )


gen_gap = (MSE_Pa_mean - MSE_a_mean/nsamples_training)/(MSE_a_mean/nsamples_training)
gen_gap_ML = (MSE_MLa_mean - MSE_a_mean/nsamples_training)/(MSE_a_mean/nsamples_training)

rel_improvement = (MSE_Pa_mean - MSE_MLa_mean)/MSE_MLa_mean

  #plot(basisA,MSE_a[i]/nsamples_training,ls='--',color=ca[i],label='Depth = ' + str(depthA[i]) )


plt.figure(1)
fig,ax = plt.subplots(1)
depthAPM_0 = np.array([1,2,3,4,5])
depthAPM=np.array([0.5,1.5,2.5,3.5,4.5,5.5])
basisAPM = np.array([5,10,15,20,25])
basisAPM=np.array([2.5,7.5,12.5,17.5, 22.5, 27.5])
depthAPM,basisAPM = np.meshgrid(depthAPM,basisAPM,indexing='ij')

pcolormesh(depthAPM,basisAPM,np.log10(MSE_a_mean),cmap='Spectral_r',edgecolors='black',)
#pcolormesh(depthAPM-0.5,basisAPM-2.5,np.log10(MSE_a_mean),cmap='Spectral_r',edgecolors='black',)
xlabel(r'Depth',**axis_font)
ylabel(r'Basis dimension',**axis_font)
cb = colorbar()
cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
ax.set_xticks(depthA )
ax.set_yticks(basisA )
savefig('MSE_training.pdf')

vmintesting = min(np.amin(np.log10(MSE_Pa_mean)),np.amin(np.log10(MSE_MLa_mean)))
vmaxtesting = max(np.amax(np.log10(MSE_Pa_mean)),np.amax(np.log10(MSE_MLa_mean)))

plt.figure(2)
fig,ax = plt.subplots(1)
pcolormesh(depthAPM-0.5,basisAPM-2.5,np.log10(MSE_Pa_mean),vmin=vmintesting,vmax=vmaxtesting,cmap='Spectral_r',edgecolors='black',)
xlabel(r'Depth',**axis_font)
ylabel(r'Basis dimension',**axis_font)
cb = colorbar()
cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
ax.set_xticks(depthA )
ax.set_yticks(basisA )
savefig('MSE_testing.pdf')

plt.figure(3)
fig,ax = plt.subplots(1)
pcolormesh(depthAPM-0.5,basisAPM-2.5,np.log10(MSE_MLa_mean),vmin=vmintesting,vmax=vmaxtesting,cmap='Spectral_r',edgecolors='black',)
xlabel(r'Depth',**axis_font)
ylabel(r'Basis dimension',**axis_font)
cb = colorbar()
cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
ax.set_xticks(depthA )
ax.set_yticks(basisA )
savefig('MSE_testing_ML.pdf')

vmingap = min(np.amin(np.log10(gen_gap)),np.amin(np.log10(gen_gap_ML)))
vmaxgap = max(np.amax(np.log10(gen_gap)),np.amax(np.log10(gen_gap_ML)))

plt.figure(4)
fig,ax = plt.subplots(1)
pcolormesh(depthAPM-0.5,basisAPM-2.5,np.log10(gen_gap),vmin=vmingap,vmax=vmaxgap,cmap='Spectral_r',edgecolors='black',)
xlabel(r'Depth',**axis_font)
ylabel(r'Basis dimension',**axis_font)
cb = colorbar()
cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
ax.set_xticks(depthA )
ax.set_yticks(basisA )
savefig('gen_gap.pdf')


plt.figure(5)
fig,ax = plt.subplots(1)
pcolormesh(depthAPM-0.5,basisAPM-2.5,np.log10(gen_gap_ML),vmin=vmingap,vmax=vmaxgap, cmap='Spectral_r',edgecolors='black',)
xlabel(r'Depth',**axis_font)
ylabel(r'Basis dimension',**axis_font)
cb = colorbar()
cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
ax.set_xticks(depthA )
ax.set_yticks(basisA )
savefig('gen_gap_ML.pdf')

plt.figure(6)
fig,ax = plt.subplots(1)
pcolormesh(depthAPM-0.5,basisAPM-2.5,rel_improvement,cmap='Spectral_r',edgecolors='black',)
xlabel(r'Depth',**axis_font)
ylabel(r'Basis dimension',**axis_font)
cb = colorbar()
cb.set_label(label=r'Relative improvement',**axis_font)
ax.set_xticks(depthA )
ax.set_yticks(basisA )
savefig('rel_improvement.pdf')


show()

#fig = plt.figure(2)
#ax = fig.gca(projection='3d')
#ax.plot_surface(depthAG,basisAG,np.log10(MSE_a),cmap=cm.jet)
#show()


'''
trainA_elu = np.zeros((nModels,np.size(models_elu[0]['train_loss'])))
for i in range(0,nModels):
  trainA_elu[i] = models_elu[i]['train_loss']

maxA_elu = np.max(trainA_elu,axis=0)
minA_elu = np.min(trainA_elu,axis=0)
epochA_elu = np.linspace(1,np.size(maxA_elu),np.size(maxA_elu))

models_box = [None]*nModels

for i in range(0,nModels):
  models_box[i] = np.load('box/stats_depth_6_nbasis_8_no_' + str(i) + '.npz')

trainA_box = np.zeros((nModels,np.size(models_box[0]['train_loss'])))
for i in range(0,nModels):
  trainA_box[i] = models_box[i]['train_loss']

maxA_box = np.max(trainA_box,axis=0)
minA_box = np.min(trainA_box,axis=0)
epochA_box = np.linspace(1,np.size(maxA_box),np.size(maxA_box))

#for i in range(0,nModels):
  #loglog(models[i]['train_loss'],color='blue')
  #loglog(models[i]['test_loss'],'--')
fill_between(epochA_elu,minA_elu/np.amax(maxA_elu),maxA_elu/np.amax(maxA_elu),color='blue',alpha=0.5,label='He')
fill_between(epochA_box,minA_box/np.amax(maxA_box),maxA_box/np.amax(maxA_box),color='red',alpha=0.5,label='Box')

xlabel('Epoch',**axis_font)
ylabel('Normalized cost',**axis_font)
yscale('log')
xscale('log')
legend(loc=1)
tight_layout()
show()
'''
