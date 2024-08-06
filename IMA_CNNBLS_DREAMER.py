import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time
import math
import scipy.io as scio


def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count / len(Label), 5))


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))


def linear(data):
    return data


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    # maximum逐个对比两个数组中元素，并选择较大的那个
    return z


def sparse_bls(A1,A2,A3,b,w11,w12,w13,w21,w22,w23,w31,w32,w33):
    lam = 0.001
    itrs = 1000
    AA1 = A1.T.dot(A1)
    AA2 = A2.T.dot(A2)
    AA3 = A3.T.dot(A3)
    
    m = A1.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk1 = x1
    ok1 = x1
    uk1 = x1
    
    wk2 = x1
    ok2 = x1
    uk2 = x1
    
    wk3 = x1
    ok3 = x1
    uk3 = x1
    
    L11 = np.mat(AA1 + np.eye(m)).I
    L12 = np.mat(AA2 + np.eye(m)).I
    L13 = np.mat(AA3 + np.eye(m)).I

    # mat可以将数组转为矩阵，I是求逆，eye返回的是一个对角线为1其他为0的矩阵
    L21 = (L11.dot(A1.T)).dot(b)
    L22 = (L12.dot(A2.T)).dot(b)
    L23 = (L13.dot(A3.T)).dot(b)
    for i in range(itrs):
        ck1 = L21 + np.dot(L11, (ok1 - uk1))
        ck2 = L22 + np.dot(L12, (ok2 - uk2))
        ck3 = L23 + np.dot(L13, (ok3 - uk3))

        ok1 = shrinkage(ck1 + uk1, lam)
        ok2 = shrinkage(ck2 + uk2, lam)
        ok3 = shrinkage(ck3 + uk3, lam)
        
        ok1 = w11*ok1 + w12*ok2 + w13*ok3
        ok2 = w21*ok1 + w22*ok2 + w23*ok3
        ok3 = w31*ok1 + w32*ok2 + w33*ok3
        

        uk1 = uk1 + ck1 - ok1
        uk2 = uk2 + ck2 - ok2
        uk3 = uk3 + ck3 - ok3
        

        wk1 = ok1
        wk2 = ok2
        wk3 = ok3
        
    wk = (wk1+wk2+wk3)/3
    
    return wk.T


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Conv_Pool(input, padding_size=2, ksize=5, stride=1, pooling_size=2, input_channel=1, out_channel=16,
              kernel_weight=None,kernel_bias=None):
    if kernel_weight is None:
        kernel_weight = random.randn(ksize * ksize * input_channel, out_channel)
    if kernel_bias is None:
        kernel_bias = random.randn(out_channel)
    len = input.shape[0]
    x = input.reshape(len, 1, 1, int(input.shape[1]))
    # padding
    time_padding = time.time()
    x_pad = np.pad(x, ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'constant')
    time_padding = time.time()-time_padding
    print("padding:", time_padding)
    # conv
    time_conv = time.time()
    N, H, W, C = x_pad.shape
    oh = (H - ksize) // stride + 1
    ow = (W - ksize) // stride + 1
    shape = (N, oh, ow, ksize, ksize, C)
    strides = (x_pad.strides[0], x_pad.strides[1] * stride, x_pad.strides[2] * stride, *x_pad.strides[1:])
    x_stride = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)# 矩阵的一个高效分块操作
    time_conv = time.time() - time_conv
    print("conv:", time_conv)


    # random intialize the weight
    kernel_weight = kernel_weight.reshape(out_channel, ksize, ksize, input_channel)
    time_forward = time.time()
    forward_path = np.einsum_path('ijk...,o...->ijko',x_stride, kernel_weight, optimize='greedy')[0]
    time_forward = time.time() - time_forward
    print("find path:", time_forward)
    time_einsum = time.time()
    x_einsum = np.einsum('ijk...,o...->ijko', x_stride, kernel_weight, optimize=forward_path)
    time_einsum = time.time() - time_einsum
    print("einsum:", time_einsum)
    time_relu = time.time()
    x_relu = relu(x_einsum)
    time_relu = time.time() - time_relu
    print("relu:", time_relu)
    time_pool = time.time()
    print(x_relu.shape)
    x_pool = x_relu.reshape(x_relu.shape[0], x_relu.shape[1] // pooling_size, pooling_size,
                            x_relu.shape[2] // pooling_size, pooling_size, x_relu.shape[3])
    x_maxpool = x_pool.max(axis=(2, 4))
    time_pool = time.time() - time_pool
    print("pool:", time_pool)
    y = x_maxpool.reshape(N, -1)
    print(y.shape)
    del kernel_weight
    return y



    # time_stride = time.time()
    # x_stride_onedimension = x_stride.reshape(x_stride.shape[0], x_stride.shape[1], x_stride.shape[2], -1)
    # time_stride = time.time() - time_stride
    # print("stride:", time_stride)

    # time_channel = time.time()
    # kernel_weight = kernel_weight.reshape(-1, kernel_weight.shape[3])
    # x_kernel = np.dot(x_stride_onedimension.reshape(-1, x_stride_onedimension.shape[3]), kernel_weight)
    # x_bias = x_kernel + kernel_bias
    # x_channel = x_bias.reshape(N, oh, ow, out_channel)
    # time_channel = time.time() - time_channel
    # print("channel:", time_channel)
    # time_relu = time.time()
    # x_relu = relu(x_channel)
    # time_relu = time.time() - time_relu
    # print("relu:", time_relu)
    # time_pool = time.time()
    # x_pool = x_relu.reshape(x_relu.shape[0], x_relu.shape[1] // pooling_size, pooling_size,
    #                    x_relu.shape[2] // pooling_size, pooling_size, x_relu.shape[3])
    # x_maxpool = x_pool.max(axis=(2, 4))
    # time_pool = time.time() - time_pool
    # print("pool:", time_pool)
    # y = x_maxpool.reshape(N, -1)
    # del kernel_weight
    # return y


def CNNBLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, conv_weight, conv_bias,w11,w12,w13,w21,w22,w23,w31,w32,w33):
    L = 0
    train_x = preprocessing.scale(train_x, axis=1)
    # scale是数据预处理，讲数据集规范化
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    # hstack是数组连接函数，ones是创建n*1的数组
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])

    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    train_acc_all = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始

    train_i = train_x
    input_channel = [1, 16]
    out_channel = [16, 32]
    kernel_size = 5
    # kernel_weight = [random.randn(kernel_size * kernel_size * input_channel[i], out_channel[i]) for i in range(N2)]
    # kernel_bias = [random.randn(out_channel[i]) for i in range(N2)]
    kernel_weight = conv_weight
    kernel_bias = conv_bias
    for i in range(N2):
        FeatureOfInputDataWithBias = np.hstack([train_i, 0.1 * np.ones((train_i.shape[0], 1))])
        random.seed(i)
        weightOfEachWindow1 = 2 * random.randn(train_i.shape[1] + 1, N1) - 1
        random.seed(i+10000)
        weightOfEachWindow2 = 2 * random.randn(train_i.shape[1] + 1, N1) - 1
        random.seed(i+20000)
        weightOfEachWindow3 = 2 * random.randn(train_i.shape[1] + 1, N1) - 1
        
        FeatureOfEachWindow1 = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow1)
        FeatureOfEachWindow2 = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow2)
        FeatureOfEachWindow3 = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow3)

        # dot是点积
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow1)
        scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow2)
        scaler3 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow3)

        FeatureOfEachWindowAfterPreprocess1 = scaler1.transform(FeatureOfEachWindow1)
        FeatureOfEachWindowAfterPreprocess2 = scaler2.transform(FeatureOfEachWindow2)
        FeatureOfEachWindowAfterPreprocess3 = scaler3.transform(FeatureOfEachWindow3)

        # 这里是一个归一化处理，将数据分布在一个范围内
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess1,FeatureOfEachWindowAfterPreprocess2,
                                      FeatureOfEachWindowAfterPreprocess3,FeatureOfInputDataWithBias,
                                      w11,w12,w13,w21,w22,w23,w31,w32,w33)
        # T是对矩阵的转置，sparse_bls求得是
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        train_i = Conv_Pool(train_i, ksize=kernel_size, input_channel=input_channel[i], out_channel=out_channel[i],
                            kernel_weight=kernel_weight[i], kernel_bias=kernel_bias[i])
        del outputOfEachWindow
        del FeatureOfEachWindow1
        del weightOfEachWindow1
        del FeatureOfEachWindow2
        del weightOfEachWindow2
        del FeatureOfEachWindow3
        del weightOfEachWindow3
        

    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)

    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer, c)
    OutputWeight = np.dot(pinvOfInput, train_y)
    time_end = time.time()
    trainTime = time_end - time_start

    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')
    train_acc_all[0][0] = trainAcc
    train_time[0][0] = trainTime
    # 测试过程
    test_x = preprocessing.scale(test_x, axis=1)
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    test_i = test_x
    time_start = time.time()

    for i in range(N2):
        FeatureOfInputDataWithBiasTest = np.hstack([test_i, 0.1 * np.ones((test_x.shape[0], 1))])
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin
        test_i = Conv_Pool(test_i, ksize=kernel_size, input_channel=input_channel[i], out_channel=out_channel[i],
                           kernel_weight=kernel_weight[i], kernel_bias=kernel_bias[i])

    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return test_acc, test_time, train_acc_all, train_time


# x = [[25 * i + j for j in range(25)] for i in range(3)]
# Conv_Pool(np.array(x))
