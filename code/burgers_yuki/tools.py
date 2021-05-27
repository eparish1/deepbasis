import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
from random import sample
from random import seed
#call when you want to make a trial or
torch.set_default_dtype(torch.float64)

class GroupMatrix:
    def __init__(self, snapshot, params, t, x, BATCH_SIZE=0):
        #ML parameters
        self.batch_size = BATCH_SIZE
        self.meshdata = 0
        self.subset = {}

        #Fom parameters
        self.snapshot = snapshot
        self.x = x
        self.t = t
        self.params = params

        #Fom normalized parameters
        self.snapshotNorm = 0
        self.xnorm =0
        self.tnorm = 0
        self.mu1norm = 0
        self.mu2norm = 0

    def createTrainMat(self):
        return GroupMatrix.TrialMatrix(self)

    def createValidateMat(self):
        return GroupMatrix.ValidationMatrix(self)

    def createDataMat(self):
        return GroupMatrix.DataMatrix(self)


    class DataMatrix:
        def __init__(self, outer_instance):
            # self.batch_size = batch_size
            self.outer_instance = outer_instance
            self.mat = 0
        def normalize(self, mat):
            self.maxInput = np.max(mat)

            self.minInput = np.min(mat)
            self.denomInput = self.maxInput-self.minInput
            self.mat = (mat-self.minInput)/self.denomInput


        def normalizev2(self,mat):
            self.mat=(mat-self.minInput)/self.denomInput


        def loadData2Pytorch(self, mat):
            # stack = torch.stack([torch.from_numpy(np.array(i)) for i in mat])
            # tensorset = torch.utils.data.TensorDataset(stack, stack)
            BATCH_SIZE = self.outer_instance.batch_size
            self.loader = \
                torch.utils.data.DataLoader(mat, batch_size=BATCH_SIZE,\
                                            shuffle=True)
        def inverseNormalize(self, mat):
            return (mat*self.denomInput)+self.minInput

    class TrialMatrix(DataMatrix):
        def __init__(self, outer_instance):#, batch_size, outer_instance, nValidate, subset):
            super().__init__( outer_instance)
            self.outer_instance = outer_instance
            #Initialize the number of elements in time, space, and parameters t
        #To be used AFTER doing meshgrid
        def createSample(self,  inputMat, nValidate):
            self.nShape = inputMat.shape
            sequence = [i for i in range(self.nShape[0])]
            self.outer_instance.subset =  np.sort(sample(sequence, nValidate))
            train_index = list(set(self.outer_instance.subset) ^ set(sequence))
            return train_index

        def createData(self, modelName, inputMat, nValidate):
            #Need to go and randomly create sample indices for each dimension
            if nValidate > 0:
                ti = self.createSample(inputMat, nValidate)
            else:
                ti = self.outer_instance.subset = np.array([])
            np.savez('train_indices_' + modelName  , indices=ti)
            index_subset=self.outer_instance.subset
            if index_subset.size>0:
                self.mat = np.delete(inputMat, index_subset, 0)
            else:
                self.mat = inputMat
            self.loadData2Pytorch(self.mat)


    class ValidationMatrix(DataMatrix):
        def __init__(self, outer_instance):
            super().__init__(outer_instance)
            self.outer_instance = outer_instance
        def createData(self, modelName, inputMat, valPercent):
            index_subset=self.outer_instance.subset
            extra =  np.sort(sample(list(index_subset), valPercent))
            np.savez('validation_indices_' + modelName  , indices=extra)
            if index_subset.size>0:
                self.mat = inputMat[extra]
                self.loadData2Pytorch(self.mat)
