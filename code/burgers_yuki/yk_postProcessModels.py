
from pylab import *
import sys
from matplotlib import rc
from deep_nn import DeepNN
import torch
import numpy as np
from driver_yuki_v2 import *
sys.path.append('/Users/yshimiz/Research/deepbasis/code/burgers_yuki/woah')
def main(argv):
    # np.set_printoptions(threshold=sys.maxsize)
    torch.set_default_dtype(torch.float64)
    axis_font = {'size':'20','family':'serif'}
    depth = np.array([int(argv[0])])
    basis = np.array([int(argv[1])])
    nEnsem = 15
    nDepth = np.size(depth)
    nBasis = np.size(basis)
    m_stats = [[[None]*15 for i in range(nBasis)] for j in range(nDepth)]
    models = [[[None]*15 for i in range(nBasis)] for j in range(nDepth)]
    #--------------------------------------------------------------------------
    # data parameters
    #--------------------------------------------------------------------------
    data_npz = np.load('snapshots_st_cn.npz') #Need to call this for nor
    snapshot = data_npz['snapshots']
    t = data_npz['t']
    x = data_npz['x']
    mu1 = data_npz['mu1']
    mu2 = data_npz['mu2']

    params = [mu1, mu2]
    data = GroupMatrix(snapshot, params, t, x)
    yk_createTrainParameter(data)
    data.snapshotNorm =  data.createDataMat()
    data.snapshotNorm.normalize(data.snapshot)
    #--------------------------------------------------------------------------
    # teseting data
    #--------------------------------------------------------------------------
    data_test_npz = np.load('snapshots_st_cn_ood.npz')
    snapshot_test = data_test_npz['snapshots']
    nx, nt, nmu = snapshot_test.shape

    t_test = data_test_npz['t']
    x_test = data_test_npz['x']
    dx = x_test[1]-x_test[0]
    mu1 = data_test_npz['mu1']
    mu2 = data_test_npz['mu2']
    nmu1 = mu1.shape[0]
    nmu2 = mu2.shape[0]
    params_test = [mu1, mu2]

    data.mu1norm.normalizev2(mu1)
    data.mu2norm.normalizev2(mu2)
    data.tnorm.normalizev2(t_test)
    data.xnorm.normalizev2(x_test)
    data.snapshotNorm.normalizev2(snapshot_test)

    in_feat=yk_createInputData(data)
    tensor_feat = torch.tensor(in_feat)

    #--------------------------------------------------------------------------
    # data parameters
    #--------------------------------------------------------------------------
    #MSE from the original runs
    MSE_a = np.zeros((nDepth, nBasis, nEnsem))
    #MSE which capsulates projection error
    MSE_Pa = np.zeros((nDepth, nBasis, nEnsem))
    #MSE comparing original to snapshot output from ML
    MSE_MLa = np.zeros((nDepth, nBasis, nEnsem))
    #--------------------------------------------------------------------------
    # blah
    #--------------------------------------------------------------------------
    s_eval=data.snapshotNorm.mat.flatten(order='F')
    for i in range(nDepth):
        for j in range(nBasis):
            for k in range(nEnsem):
                d_i = str(depth[i])
                b_i = str(basis[j])
                e_k = str(k)
                file_nm = 'stats_dropout_depth_'+ d_i +'_nbasis_'+b_i+'_no_'+e_k+'.npz'
                file_nm2 = 'dropout_depth_'+d_i+'_nbasis_'+b_i+'_no_'+e_k+'_trained'
                file_ld = torch.load(file_nm2, map_location='cpu')
                m_stats[i][j][k] = np.load(file_nm)
                models[i][j][k] = DeepNN(depth[i], basis[j], False)
                models[i][j][k].load_state_dict(file_ld)
                models[i][j][k].to('cpu')
                models[i][j][k].eval()
                Phi = models[i][j][k].createBasis(tensor_feat).detach().numpy()
                Phi,_=np.linalg.qr(Phi, mode='reduced')
                PhiT = np.transpose(Phi)

                uml = models[i][j][k](tensor_feat)
                uml = uml.detach().numpy()
                ufom_p = Phi @ (PhiT @ s_eval)
                # uform_a = m_stats[i][j][k]['train_loss'][-1]
                np.savez('ood_dropout_test_projected_depth_'+ d_i +'_nbasis_'+b_i+'_no_'+e_k+'.npz', states = ufom_p)
                np.savez('ood_dropout_test_ml_depth_'+ d_i +'_nbasis_'+b_i+'_no_'+e_k+'.npz', states = uml[:,0])

                #uform_a = data.snapshotNorm.inverseNormalize(uform_a)
                # MSE_Pa[i,j,k] = np.mean( (ufom_p - s_eval)**2 )
                # MSE_MLa[i,j,k] = np.mean( (uml[:,0] - s_eval)**2 )
                # MSE_a[i,j,k] = uform_a

                # ss = np.reshape(ufom_p, (nmu,nt, nx))
                # ss = data.snapshotNorm.inverseNormalize(ss)
                # plt.figure(1)
                # for t in {100,150,200,250,300,350}:
                #      plt.plot(x, snapshot_test[:,t,0],'b')
                #      plt.plot(x, ss[0,t,:],'r')

    # MSE_Pa_mean = np.mean(MSE_Pa,axis=2)
    # MSE_MLa_mean = np.mean(MSE_MLa,axis=2)
    # print(MSE_MLa_mean)
    # MSE_a_mean = np.mean(MSE_a,axis=2)
    # print(MSE_Pa_mean, MSE_MLa_mean, MSE_a_mean)
    # depth1=np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
    # basis1=np.array([5,10,15,20,25,30,35,40,45,50,55])
    # depth1 ,basis1 = np.meshgrid(depth1, basis1,indexing='ij')

    # plt.savefig('trajectories.pdf')
    #figure(1)
    #fig,ax = plt.subplots(1)
    #pcolormesh(depth1,basis1,np.log10(MSE_Pa_mean),cmap='Spectral_r',edgecolors='black',)
    #xlabel(r'Depth',**axis_font)
    #ylabel(r'Basis dimension',**axis_font)
    #cb = colorbar()
    #cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
    #plt.savefig('MSE_Pa_mean.pdf')

#    ax.set_xticks(depth)
 #   ax.set_yticks(basis)

    #figure(2)
    #fig,ax=plt.subplots(1)
    #pcolormesh(depth1,basis1,np.log10(MSE_MLa_mean),cmap='Spectral_r',edgecolors='black',)
    #xlabel(r'Depth',**axis_font)
    #ylabel(r'Basis dimension',**axis_font)
    #cb = colorbar()
    #cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
    #plt.savefig('MSE_MLA_mean.pdf')

  #  ax.set_xticks(depth)
  #  ax.set_yticks(basis)

    #figure(3)
    #fig,ax=plt.subplots(1)
    #pcolormesh(depth1,basis1,np.log10(MSE_a_mean),cmap='Spectral_r',edgecolors='black',)
    #xlabel(r'Depth',**axis_font)
    #ylabel(r'Basis dimension',**axis_font)
    #cb = colorbar()
    #cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
    #plt.savefig('MSE_a_mean.pdf')

# ax.set_xticks(depth)
   # ax.set_yticks(basis)

    # show()
if __name__ == '__main__':
    main(sys.argv[1:])
