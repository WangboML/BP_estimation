# -*-coding:utf-8-*-

import random
import numpy as np
import os
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib
import math
import pandas as pd


import matplotlib as mpl

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
#xmajorLocator   = MultipleLocator(1) #将x主刻度标签设置为20的倍数

#chinese_font = mpl.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
# 首先将所有的特征数据进行输入



def input_data_pca(data_path,data,train_rate1):

    #     # 首先定义每一个
    datalist_path=os.path.join(data_path,data)
    data_list = sio.loadmat(datalist_path)

    PPG_Features=data_list['PPG_Features'].tolist()
    ECG_Features=data_list['ECG_Features'].tolist()
    PPG_ECG_alingFeatures=data_list['PPG_ECG_alingFeatures'].tolist()
    DIA_BP=data_list['DIA_BP'].tolist()
    SYS_BP=data_list['SYS_BP'].tolist()
    X_Features=[]
    for a,b,c in zip(PPG_Features, ECG_Features,PPG_ECG_alingFeatures):
        l1=[]
        l1.extend(a)
        l1.extend(b)
        l1.extend(c)
        X_Features.append(l1)

    # 对每一组数据划分为测试集和训练集，最终将所有的测试集和训练集拿出来，得到一个综合的测试集和训练集

    length=len(X_Features)
    train_length=math.ceil(train_rate1*length)
    X_train=X_Features[0:train_length]
    X_test=X_Features[train_length:]
    DIA_BP_train=DIA_BP[0][0:train_length]
    DIA_BP_test=DIA_BP[0][train_length:]
    SYS_BP_train = SYS_BP[0][0:train_length]
    SYS_BP_test = SYS_BP[0][train_length:]
    #plt.plot(DIA_BP_test,'b-*')


    # X的标准化，放到[0,1]区间内，这里的min_max_scaler_DIA、min_max_scaler_SYS可能要保留
    standard_scaler_X = preprocessing.StandardScaler()
    X_train_scaler = standard_scaler_X.fit_transform(X_train)
    X_test_scaler = standard_scaler_X.transform(X_test)
    # 2 Y的标准化，放到[0,1]区间内，这里的min_max_scaler_DIA、min_max_scaler_SYS可能要保留
    # 由于DIA和SYS是一维行列表

    DIA_TRAIN_array=np.array(DIA_BP_train)
    DIA_TRAIN_tra=DIA_TRAIN_array.reshape(-1,1)
    DIA_TEST_array = np.array(DIA_BP_test)
    DIA_TEST_tra = DIA_TEST_array.reshape(-1, 1)
    standard_scaler_DIA = preprocessing.MinMaxScaler()
    DIA_train_minmax = standard_scaler_DIA.fit_transform(DIA_TRAIN_tra)
    DIA_test_minmax = standard_scaler_DIA.transform(DIA_TEST_tra)




    SYS_TRAIN_array = np.array(SYS_BP_train)
    SYS_TRAIN_tra = SYS_TRAIN_array.reshape(-1, 1)
    SYS_TEST_array = np.array(SYS_BP_test)
    SYS_TEST_tra = SYS_TEST_array.reshape(-1, 1)
    standard_scaler_SYS = preprocessing.MinMaxScaler()
    SYS_train_minmax = standard_scaler_SYS.fit_transform(SYS_TRAIN_tra)
    SYS_test_minmax = standard_scaler_SYS.transform(SYS_TEST_tra)

    # # 找出训练集特征向量的方差贡献率
    # # 首先计算协方差矩阵,并且画出方差贡献图
    # cov_mat=np.cov(X_train_minmax.T)
    # eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
    # print(eigen_vals)
    # tot=sum(eigen_vals)
    # var_exp=[(i/tot) for i in sorted(eigen_vals,reverse=True)]
    # cum_var_exp=np.cumsum(var_exp)
    # plt.bar(range(1,40),var_exp,alpha=0.5,align='center',label=u'单个方差贡献')
    # plt.step(range(1, 40), cum_var_exp, where='mid',  label=u'累计特征方差贡献')
    # plt.ylabel(u'方差贡献率')
    # plt.xlabel(u'主成分')
    # plt.legend(loc='best')
    # plt.show()

    # 进行pca降低唯独，使用15个
    pca=PCA(n_components=15)
    X_train_pca=pca.fit_transform(X_train_scaler)
    X_test_pca=pca.transform(X_test_scaler)

    standard_scaler_features = preprocessing.MinMaxScaler()
    X_train_pca = standard_scaler_features.fit_transform(X_train_pca)
    X_test_pca = standard_scaler_features.transform(X_test_pca)


    # 将得到的所有数据形成字典进行保存，之后再永久保存
    train_data={'features':X_train_pca,'sys_bp':SYS_train_minmax,'dia_bp':DIA_train_minmax}

    test_data ={'features': X_test_pca, 'sys_bp': SYS_test_minmax, 'dia_bp': DIA_test_minmax}

    frame = pd.DataFrame(X_train_pca)
    #print(frame)
    sys_train=pd.DataFrame(SYS_train_minmax.reshape(-1))
    print(sys_train)
    dia_train=pd.DataFrame(DIA_train_minmax.reshape(-1))
    persion_sys =frame.corrwith(sys_train[0])
    persion_dia = frame.corrwith(dia_train[0])
    per_sys = [abs(i)  for i in sorted(persion_sys, reverse=True)]
    per_dia = [abs(i) for i in sorted(persion_dia, reverse=True)]
    per_sys = [abs(i) for i in sorted(per_sys, reverse=True)]
    per_dia = [abs(i) for i in sorted(per_dia, reverse=True)]
    print(per_sys)
    plt.plot(np.linspace(1,15,15),per_sys,'cv--',markersize=20)

    # plt.step(range(1, 40), cum_var_exp, where='mid',  label=u'累计特征方差贡献')
    plt.ylabel(u'特征与SYS的相关系数')
    plt.xlabel(u'PCA特征序号')
    plt.legend(loc='best')
    #plt.xlim(1,15)
    #ax = plt.gca()
    plt.xticks(np.linspace(1,15,15))
    #ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12','13','14','15'))
    plt.show()



    return None





def main():
    pass
    data_path = '/home/wangbo/bp/features_data/'
    data = 'data8.mat'
    train_rate1 = 0.02
    input_data_pca(data_path, data, train_rate1)


if __name__ == '__main__':
    main()
