import numpy as np
import scipy.io as scio
from IMA_CNNBLS_DREAMER import CNNBLS
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import normalize
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as Data
import torchvision  # torchvision模块包括了一些图像数据集,如MNIST,cifar10等
import matplotlib.pyplot as plt
import time
import CNNet_DREAMER
import numpy
from torch.utils.data import TensorDataset
from keras.utils import to_categorical
''' For Keras dataset_load()'''
# import keras 
# (traindata, trainlabel), (testdata, testlabel) = keras.datasets.mnist.load_data()
# traindata = traindata.reshape(traindata.shape[0], 28*28).astype('float64')/255
# trainlabel = keras.utils.to_categorical(trainlabel, 10)
# testdata = testdata.reshape(testdata.shape[0], 28*28).astype('float64')/255
# testlabel = keras.utils.to_categorical(testlabel, 10)


data_training = np.load('/Volumes/T7 Shield/数据集/model_5_Khosru2/data_training.npy')
data_testing = np.load('/Volumes/T7 Shield/数据集/model_5_Khosru2/data_testing.npy')
label_training = np.load('/Volumes/T7 Shield/数据集/model_5_Khosru2/label_training.npy')
label_testing = np.load('/Volumes/T7 Shield/数据集/model_5_Khosru2/label_testing.npy')
#a=scipy.io.loadmat('/Volumes/T7 Shield/数据集/DREAMER.mat')['DREAMER']
x_train = normalize(data_training)
x_test = normalize(data_testing)

#1:Valence;0:Arousal
y_train_valence0 = np.ravel(label_training[:, [1]])
y_test_valence0 = np.ravel(label_testing[:, [1]])
y_train_arousal0 = np.ravel(label_training[:,[0]])
y_test_arousal0 = np.ravel(label_testing[:, [0]])
# Arousal_Train = np.ravel(Y[:, [0]])
# Valence_Train = np.ravel(Y[:, [1]])
# Domain_Train = np.ravel(Y[:, [2]])
# Like_Train = np.ravel(Y[:, [3]])

y_train_valence = []
for i in range(len(y_train_valence0)):#40
    if y_train_valence0[i] <= 5:
        y_train_valence.append(0)
    else:                 
        y_train_valence.append(1)
y_train_valence = np.array(y_train_valence)
# print(valence_labels)
y_test_valence = []
for i in range(len(y_test_valence0)):
    if y_test_valence0[i] <= 5:
        y_test_valence.append(0)
    else:               
        y_test_valence.append(1)
y_test_valence = np.array(y_test_valence)
y_train_valence = to_categorical(y_train_valence)
y_test_valence = to_categorical(y_test_valence)

cnn = CNNet_DREAMER.CNN()
cnn.load_state_dict(torch.load('model/cnn_dreamer.pth'))
weight = cnn.state_dict()
conv1_weight = np.array(weight['conv1.weight'])
conv2_weight = np.array(weight['conv2.weight'])
conv_weight = [conv1_weight,conv2_weight]

conv1_bias = np.array(weight['conv1.bias'])
conv2_bias = np.array(weight['conv2.bias'])
conv_bias = [conv1_bias,conv2_bias]

N1 = 20  #  # of nodes belong to each window
N2 = 2 #  # of windows -------Feature mapping layer
N3 = 6000 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps 
M1 = 50  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

print('-------------------CNNBLS_BASE---------------------------')
CNNBLS(x_train, y_train_valence, x_test, y_test_valence, s, C, N1, N2, N3,conv_weight,conv_bias,0.8,0.1,0.1,0.04,0.9,0.06,0.02,0.03,0.95)
# print('-------------------BLS_BASE---------------------------')
# BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
# print('-------------------BLS_ENHANCE------------------------')
# BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
# print('-------------------BLS_FEATURE&ENHANCE----------------')
# M2 = 50  #  # of adding feature mapping nodes
# M3 = 50  #  # of adding enhance nodes
# BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)













'''
teA = list() #Testing ACC 
tet = list() #Testing Time
trA = list() #Training ACC
trt = list() #Training Time
t0 = 0
t2 =[]
t1 = 0
tt1 = 0
tt2 = 0
tt3 = 0
# BLS parameters
s = 0.8  #reduce coefficient
C = 2**(-30) #Regularization coefficient
N1 = 22  #Nodes for each feature mapping layer window 
N2 = 20  #Windows for feature mapping layer
N3 = 540 #Enhancement layer nodes
#  bls-网格搜索
for N1 in range(8,25,2):
    r1 = len(range(8,25,2))
    for N2 in range(10,21,2):
        r2 = len(range(10,21,2))
        for N3 in range(600,701,10):
            r3 = len(range(600,701,10))
            a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3)
            t0 += 1
            if a>t1:
                tt1 = N1
                tt2 = N2
                tt3 = N3
                t1 = a
            teA.append(a)
            tet.append(b)
            trA.append(c)
            trt.append(d)
            print('percent:' ,round(t0/(r1*r2*r3)*100,4),'%','The best result:', t1,'N1:',tt1,'N2:',tt2,'N3:',tt3)
meanTeACC = np.mean(teA)
meanTrTime = np.mean(trt)
maxTeACC = np.max(teA)   
'''
'''
#BLS随机种子搜索
teA = list() #Testing ACC 
tet = list() #Testing Time
trA = list() #Training ACC
trt = list() #Train Time
t0 = 0
t = 0
t2 =[]
t1 = 0
tt1 = 0
tt2 = 0
tt3 = 0
## BLS parameters
s = 0.8 #reduce coefficient
C = 2**(-30) #Regularization coefficient
#N1 = 10  #Nodes for each feature mapping layer window 
#N2 = 10  #Windows for feature mapping layer
#N3 = 500 #Enhancement layer nodes
dataFile = './/dataset//mnist.mat'
data = scio.loadmat(dataFile)
traindata,trainlabel,testdata,testlabel = np.double(data['train_x']/255),2*np.double(data['train_y'])-1,np.double(data['test_x']/255),2*np.double(data['test_y'])-1
u = 45
i = 0
L = 5
M = 50
for N1 in range(10,21,20):
    r1 = len(range(10,21,20))
    for N2 in range(12,21,10):
        r2 = len(range(12,21,10))
        for N3 in range(4000,4001,500):
            r3 = len(range(4000,4001,500))
            for i in range(-28,-27,2):
                r4 = len(range(-28,-27,2))
                C = 2**(i)
#                a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,u)
                a,b,c,d = BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M)
#                t0 += 1
#                if a>t1:
#                    tt1 = N1
#                    tt2 = N2
#                    tt3 = N3
#                    t1 = a
#                    t = u
#                    i1 = i
#                tet.append(b)    
#                teA.append(a)               
#                trA.append(c)
#                trt.append(d)
#                print('NO.',t0,'total:',r1*r2*r3,'ACC:',a*100,'Pars:',N1,',',N2,',',N3,'C',i)
#                print('The best so far:', t1*100,'N1:',tt1,'N2:',tt2,'N3:',tt3,'C:',i1)
                print('working ...')
                print('teACC',teA,'teTime',tet,'trACC',trA,'trTime',trt)
'''
#Grid search for Regularization coefficient 

#
'''
teA = list()
tet = list()
trA = list()
trt = list()
s = 0.8
#C = 2**(-30)
N1 = 10
N2 = 100
N3 = 8000
L = 5
M1 = 20
M2 = 20
M3 = 50
t0 = 0
t1 = 0
t2 = 0
for i in range(-30,-21,5):
    r1 = len(range(-30,-21,5))
    for u in range(10,50,1):
        r2 = len(range(10,50,1))
        C = 2**i
        t0 += 1 
#    a,b,c,d = BLS_AddEnhanceNodes(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)
        a,b,c,d = BLS(traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,u)
        teA.append(a)
        tet.append(b)
        trA.append(c)
        trt.append(d)
        if a > t1:
            t1 = a
            t2 = i
            t = u
        print(t0,'percent:',round(t0/(r1*r2)*100,4),'%','The best result:', t1,'C',t2,'u:',t)
'''     








