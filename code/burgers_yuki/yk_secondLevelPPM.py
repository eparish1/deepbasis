#from pylab import *
import sys
from matplotlib import rc
from deep_nn import DeepNN
import torch
import numpy as np
from driver_yuki_v2 import *
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    os.environ["PATH"]+= os.pathsep+ '/opt/spack/c7.7/stock-20201208/spack/opt/spack/linux-centos7-x86_64/gcc-7.5.0/texlive-live-fieigbiggeh6tzixv4hvqzgyaiswpnca/bin/x86_64-linux'
    # os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2018/bin/x86_64-darwin'
    mpl.use('Agg')
    rc('text', usetex=True)
    plt.rcParams["text.usetex"] = False
    rc('text.latex', preamble=r'\usepackage{amsmath},\usepackage{amssymb}')
    rc('font', size=12)
    # axis_font = {'size':10,'family':'serif'}
    mpl.rc('font',family='serif')
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.dashed_pattern'] = [6, 6]
    mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
    mpl.rcParams['lines.dotted_pattern'] = [1, 3]
    mpl.rcParams['lines.scale_dashes'] = False
    torch.set_default_dtype(torch.float64)
    axis_font = {'size':'12','family':'serif'}
    basis = np.array([1,2,4,8,16,24,32,40,48,56,64,80,96])
    depth = np.array([1,2,3,4,5,6,7])
    # basis = np.array([48])
    # depth = np.array([6])
    nEnsem = 15
    nDepth = np.size(depth)
    nBasis = np.size(basis)

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

    data_test_npz = np.load('snapshots_st_cn_id.npz')
    snapshot_test = data_test_npz['snapshots']
    data.snapshotNorm.normalizev2(snapshot_test)
    s_eval=data.snapshotNorm.mat.flatten(order='F')
    #MSE which capsulates projection error
    MSE_Pa_id = np.zeros((nDepth, nBasis, nEnsem))
    #MSE comparing original to snapshot output from ML
    MSE_MLa_id = np.zeros((nDepth, nBasis, nEnsem))
    for i in range(nDepth):
        for j in range(nBasis):
            for k in range(nEnsem):
                d_i = str(depth[i])
                b_i = str(basis[j])
                e_k = str(k)
                u_proj = np.load('id_dropout_test_projected_depth_'+ d_i +'_nbasis_'+b_i+'_no_'+e_k+'.npz')['states']
                uml=np.load('id_dropout_test_ml_depth_'+ d_i +'_nbasis_'+b_i+'_no_'+e_k+'.npz')['states']
                MSE_Pa_id[i,j,k] = np.mean( (u_proj - s_eval)**2 )
                MSE_MLa_id[i,j,k] = np.mean( (uml - s_eval)**2 )
    MSE_Pa_mean_id = np.mean(MSE_Pa_id,axis=2)
    MSE_MLa_mean_id = np.mean(MSE_MLa_id,axis=2)

    depth1=np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
    basis1=np.array([2.5,7.5,12.5,17.5,22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5,67.5])
    # depth1 = np.array([0.5, 1.5])
    # basis1 = np.array([5,10])
    depth1 ,basis1 = np.meshgrid(depth1, basis1,indexing='ij')

    basis_tmp = np.array([5,10,15,20,25,30,35,40,45,50])

    data_test_npz = np.load('snapshots_st_cn_ood.npz')
    snapshot_test = data_test_npz['snapshots']
    data.snapshotNorm.normalizev2(snapshot_test)
    s_eval=data.snapshotNorm.mat.flatten(order='F')
    #MSE which capsulates projection error
    MSE_Pa_ood = np.zeros((nDepth, nBasis, nEnsem))
    #MSE comparing original to snapshot output from ML
    MSE_MLa_ood = np.zeros((nDepth, nBasis, nEnsem))
    for i in range(nDepth):
        for j in range(nBasis):
            for k in range(nEnsem):
                d_i = str(depth[i])
                b_i = str(basis[j])
                e_k = str(k)
                u_proj = np.load('ood_dropout_test_projected_depth_'+ d_i +'_nbasis_'+b_i+'_no_'+e_k+'.npz')['states']
                uml=np.load('ood_dropout_test_ml_depth_'+ d_i +'_nbasis_'+b_i+'_no_'+e_k+'.npz')['states']
                MSE_Pa_ood[i,j,k] = np.mean( (u_proj - s_eval)**2 )
                MSE_MLa_ood[i,j,k] = np.mean( (uml - s_eval)**2 )

    MSE_Pa_mean_ood = np.mean(MSE_Pa_ood,axis=2)
    MSE_MLa_mean_ood = np.mean(MSE_MLa_ood,axis=2)

    MSE_a = np.zeros((nDepth, nBasis, nEnsem))
    MSE_a_P = np.zeros((nDepth, nBasis, nEnsem))

    for i in range(nDepth):
        for j in range(nBasis):
            for k in range(nEnsem):
                d_i = str(depth[i])
                b_i = str(basis[j])
                e_k = str(k)
                file_nm = 'stats_dropout_depth_'+ d_i+ '_nbasis_'+b_i+'_no_'+e_k+'.npz'
                file_p = 'train_projectedrrror_dropout_depth_'+ d_i +'_nbasis_'+b_i+'_no_'+e_k+'.npz'

                uform_a = np.load(file_nm)['train_loss'][-1]
                upom = np.load(file_p)['mse']
                MSE_a[i,j,k] = uform_a
                MSE_a_P[i,j,k] = upom
    MSE_a_mean = np.mean(MSE_a, axis=2)
    MSE_a_P_mean = np.mean(MSE_a_P, axis=2)
    depth1=np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
    basis1=np.array([2.5,7.5,12.5,17.5,22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5,67.5])
    # depth1 = np.array([0.5, 1.5])
    # basis1 = np.array([5,10])
    depth1 ,basis1 = np.meshgrid(depth1, basis1,indexing='ij')

    basis_tmp = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65])

    vvmin = np.amin(np.array([MSE_a_mean.min(), MSE_a_P_mean.min()]))
    vvmax = np.amax(np.array([MSE_a_mean.max(), MSE_a_P_mean.max()]))

    vvmin_id = np.amin(np.array([MSE_Pa_mean_id.min(), MSE_MLa_mean_id.min()]))
    vvmax_id = np.amax(np.array([MSE_Pa_mean_id.max(), MSE_MLa_mean_id.max()]))

    vvmin_ood = np.amin(np.array([MSE_Pa_mean_ood.min(), MSE_MLa_mean_ood.min()]))
    vvmax_ood = np.amax(np.array([MSE_Pa_mean_ood.max(), MSE_MLa_mean_ood.max()]))

    ult_min = np.min(np.array([vvmin_id, vvmin_ood, vvmin]))
    ult_max = np.max(np.array([vvmax_id, vvmax_ood, vvmax]))
    plt.figure(1)
    fig,ax = plt.subplots(1)
    im=ax.pcolormesh(depth1,basis1,MSE_Pa_mean_id,norm=colors.LogNorm(vmin=ult_min, vmax=ult_max),cmap='Spectral_r',edgecolors='black', linewidths=0.5)
    plt.xlabel(r'depth',**axis_font)
    # plt.ylabel(r'basis dimension',**axis_font)
    original_loc = ax.get_axes_locator()
    cb=plt.colorbar(im)
    cb.ax.tick_params(axis='y',  which='both',direction='in', pad=3)
    ax.set_xticks(depth)
    # ax.set_yticks(basis_tmp)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='both', direction='in', pad = 4)
    ax.tick_params(axis='y', which='both', direction='in', length=0,pad = 4)
    plt.gcf().set_size_inches(6, 4.84)
    cb.remove()
    ax.set_axes_locator(original_loc)
    plt.savefig('dropout_mean_MSE_Pa_mean_id.pdf', bbox_inches = 'tight', format='pdf', pad_inches=0.01)

    plt.figure(2)
    fig,ax = plt.subplots(1)
    im=ax.pcolormesh(depth1,basis1,MSE_MLa_mean_id,norm=colors.LogNorm(vmin=ult_min, vmax=ult_max),cmap='Spectral_r',edgecolors='black', linewidths=0.5)
    plt.xlabel(r'depth',**axis_font)
    original_loc = ax.get_axes_locator()
    # plt.ylabel(r'basis dimension',**axis_font)
    cb=plt.colorbar(im)
    cb.ax.tick_params(axis='y',  which='both',direction='in', pad=3)
    ax.set_xticks(depth)
    # ax.set_yticks(basis_tmp)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='both', direction='in', pad = 4)
    ax.tick_params(axis='y', which='both', direction='in', length=0,pad = 4)
    plt.gcf().set_size_inches(6, 4.84)
    cb.remove()
    ax.set_axes_locator(original_loc)
    plt.savefig('dropout_mean_MSE_MLA_mean_id.pdf', bbox_inches = 'tight', format='pdf', pad_inches=0.01)

    plt.figure(3)
    fig,ax = plt.subplots(1)
    im=ax.pcolormesh(depth1,basis1,MSE_Pa_mean_ood,norm=colors.LogNorm(vmin=ult_min, vmax=ult_max),cmap='Spectral_r',edgecolors='black', linewidths=0.5)
    plt.xlabel(r'depth',**axis_font)
    # plt.ylabel(r'basis dimension',**axis_font)
    cb=plt.colorbar(im)
    cb.ax.tick_params(axis='y',  which='both',direction='in', pad=3)
    ax.set_xticks(depth)
    # ax.set_yticks(basis_tmp)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='both', direction='in', pad = 4)
    ax.tick_params(axis='y', which='both', direction='in', length=0,pad = 4)
    plt.gcf().set_size_inches(6, 4.84)
    plt.savefig('dropout_mean_MSE_Pa_mean_ood.pdf', bbox_inches = 'tight', format='pdf', pad_inches=0.01)

    plt.figure(4)
    fig,ax = plt.subplots(1)
    im=ax.pcolormesh(depth1,basis1,MSE_MLa_mean_ood,norm=colors.LogNorm(vmin=ult_min, vmax=ult_max),cmap='Spectral_r',edgecolors='black', linewidths=0.5)
    plt.xlabel(r'depth',**axis_font)
    # plt.ylabel(r'basis dimension',**axis_font)
    cb=plt.colorbar(im)
    cb.ax.tick_params(axis='y',  which='both',direction='in', pad=3)
    ax.set_xticks(depth)
    # ax.set_yticks(basis_tmp)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='both', direction='in', pad = 4)
    ax.tick_params(axis='y', which='both', direction='in', length=0,pad = 4)
    plt.gcf().set_size_inches(6, 4.84)
    plt.savefig('dropout_mean_MSE_MLA_mean_ood.pdf', bbox_inches = 'tight', format='pdf', pad_inches=0.01)

    plt.figure(5)
    fig,ax = plt.subplots(1)
    im=ax.pcolormesh(depth1,basis1,MSE_a_mean,norm=colors.LogNorm(vmin=ult_min, vmax=ult_max),cmap='Spectral_r',edgecolors='black', linewidths=0.5)
    plt.xlabel(r'depth',**axis_font)
    original_loc = ax.get_axes_locator()
    plt.ylabel(r'basis dimension',**axis_font)
    cb=plt.colorbar(im)
    cb.ax.tick_params(axis='y',  which='both',direction='in', pad=3)
    ax.set_xticks(depth)
    ax.set_yticks(basis_tmp)
    ax.set_yticklabels(basis)
    ax.tick_params(axis='x', which='both', direction='in', pad = 4)
    ax.tick_params(axis='y', which='both', direction='in', pad = 4)
    plt.gcf().set_size_inches(6, 4.84)
    cb.remove()
    ax.set_axes_locator(original_loc)
    plt.savefig('dropout_mean_MSE_a_mean.pdf', bbox_inches = 'tight', format='pdf', pad_inches=0.01)

    plt.figure(6)
    fig,ax = plt.subplots(1)
    im=ax.pcolormesh(depth1,basis1,MSE_a_P_mean,norm=colors.LogNorm(vmin=ult_min, vmax=ult_max),cmap='Spectral_r',edgecolors='black', linewidths=0.5)
    plt.xlabel(r'depth',**axis_font)
    original_loc = ax.get_axes_locator()
    plt.ylabel(r'basis dimension',**axis_font)
    cb=plt.colorbar(im)
    cb.ax.tick_params(axis='y',  which='both',direction='in', pad=3)
    ax.set_xticks(depth)
    ax.set_yticks(basis_tmp)
    ax.set_yticklabels(basis)
    ax.tick_params(axis='x', which='both', direction='in', pad = 4)
    ax.tick_params(axis='y', which='both', direction='in', pad = 4)
    plt.gcf().set_size_inches(6, 4.84)
    cb.remove()
    ax.set_axes_locator(original_loc)
    plt.savefig('dropout_mean_MSE_a_P_mean.pdf', bbox_inches = 'tight', format='pdf', pad_inches=0.01)


    # fig,ax=plt.subplots(1)
    # im = ax.pcolormesh(depth1,basis1,np.log10(MSE_MLa_mean),cmap='Spectral_r',edgecolors='black',)
    # plt.xlabel(r'Depth',**axis_font)
    # plt.ylabel(r'Basis dimension',**axis_font)
    # cb = fig.colorbar(im)
    # # cb.set_label(label=r'$\log_{10}$(MSE)',**axis_font)
    # plt.savefig('MSE_MLA_mean.pdf')
