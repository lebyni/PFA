from __future__ import print_function
import torch
import math
import random
# import seaborn as sns
import heapq
import numpy as np
# from matplotlib import pyplot as plt
import scipy.stats
from scipy.stats import norm, stats
from torchvision import models
import collections

'''异常检测主运行程序'''


def anomalyDetection_example():

    '''加载并显示数据'''

    sim_list=[]
    whole_list=[]
    # total_list=np.zeros((1,8))
    path_dir = []
    test_path = 'result_sparsity_8client/client'
    # test_path = 'result_mobilenet/client'

    for i in range(8):
        tardir = test_path + str(i) + '_layer27_ratio_office'
        path_dir.append(tardir)

    # print(path_dir)
    # exit()

    for index in range(8):
        target = np.loadtxt('result_sparsity_8client/client0_layer27_ratio_office', dtype=np.float32)
        # target = np.loadtxt('result_mobilenet/client0_layer2_ratio_mobilenet', dtype=np.float32)
        test = np.loadtxt(path_dir[index], dtype=np.float32)
        channel_num = 20
        # print(target.shape)
        # print(test1.shape)
        sample_mean = np.mean(target, axis=0)
        sample_mean = sample_mean.tolist()
        max_num_index_mean = list(map(sample_mean.index, heapq.nlargest(channel_num, sample_mean)))
        # sample_var = np.var(target, axis=0)
        # sample_var = sample_var.tolist()
        # min_num_index_var = list(map(sample_var.index, heapq.nsmallest(channel_num, sample_var)))
        random.seed(102)
        index_list_bn = random.sample(range(0, 512), channel_num)
        # index_list_bn = max_num_index_mean
        target = target[:, index_list_bn]
        test = test[:, index_list_bn]

        target_avg=np.mean(target,axis=0)
        test_avg = np.mean(test, axis=0)
        # print(target_avg.shape)
        # print(target_avg)
        # print(test_avg)

        #Euclidean Distance
        dist = np.sqrt(np.sum(np.square(target_avg - test_avg)))
        sim_list.append(dist)
        # sim=cos_sim(target_avg,test_avg)
        # print(dist,sim)
    whole_list.append(sim_list)
    temp = np.array(whole_list)
    f=np.loadtxt('../plot/neuron_location/20neuron')
    # f=f[np.newaxis,:]
    print(temp.shape)
    print(f.shape)
    f=np.concatenate((f, temp), axis=0)
    # f=np.delete(f, 0, 1)

    np.savetxt('../plot/neuron_location/20neuron',f)





def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def calculate_gaussian_kl_divergence(m1,m2,v1,v2):
    ###m1,m2 指两个高斯分布的均值
    ###v1，v2指两个高斯分布的方差
    return np.log(v2 / v1) + (v1*v1+(m1-m2)*(m1-m2))/(2*v2*v2) - 0.5


def compute_cov(mu,X):
    dim=X[0].size
    # print(dim)
    mu = np.mean(X, axis=0)
    for i in range(X[:,0].size):
        X[i]=X[i]-mu
    sigma2=(1/dim)*np.dot(X.T,X)
    return sigma2

def estimateGaussian(X):
    m, n = X.shape
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))
    mu = np.mean(X, axis=0)  # axis=0表示列，每列的均值
    var=np.var(X,axis=0)
    # sigma2 = np.cov(X.T)  # 求每列的协方差
    return mu, var

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s

# 高斯分布函数

def norm_gaussian(dataset, mu, sigma):
    # p = norm(mu, sigma)
    # return p.pdf(dataset)
    return norm(mu, sigma).pdf(dataset)


if __name__ == '__main__':
    anomalyDetection_example()
