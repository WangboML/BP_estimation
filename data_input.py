# -*-coding:utf-8-*-

import random
import numpy as np
import os
import scipy.io as sio
from sklearn import preprocessing

from get_parser import FLAGS
import math
import pickle

# 首先将所有的特征数据进行输入



def input_data(data_path,data,train_rate1):

    # if data_update:
        # data_dir = os.listdir(data_path)
        # # 从中随机的选择10个作为模型的主要训练集合，其中对每一个数据集0.8作为训练，0.2作为测试
        # # data_dir1 = random.sample(data_dir, 10) # 下面就是这个随机得到的
        # data_dir1 = ['realdata1.mat', 'data6.mat', 'realdata3.mat', 'realdata7.mat', 'data10.mat', 'data4.mat',
        #              'data13.mat', 'data7.mat', 'data18.mat', 'realdata4.mat']
        # data_dir2 = []
        # for list in data_dir:
        #     if list not in data_dir1:
        #         data_dir2.append(list)
        #
        # # 对所有的用于模型参数训练的数据进行处理
        # # 将所有的.mat文件转换成列表形式
        # X_TRAIN=[]
        # X_TEST=[]
        # DIA_TRAIN=[]
        # DIA_TEST=[]
        # SYS_TRAIN=[]
        # SYS_TEST=[]
        # for data in data_dir1:
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



    # X的标准化，放到[0,1]区间内，这里的min_max_scaler_DIA、min_max_scaler_SYS可能要保留
    min_max_scaler_X = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler_X.fit_transform(X_train)
    X_test_minmax = min_max_scaler_X.transform(X_test)
    # 2 Y的标准化，放到[0,1]区间内，这里的min_max_scaler_DIA、min_max_scaler_SYS可能要保留
    # 由于DIA和SYS是一维行列表

    DIA_TRAIN_array=np.array(DIA_BP_train)
    DIA_TRAIN_tra=DIA_TRAIN_array.reshape(-1,1)
    DIA_TEST_array = np.array(DIA_BP_test)
    DIA_TEST_tra = DIA_TEST_array.reshape(-1, 1)
    min_max_scaler_DIA = preprocessing.MinMaxScaler()
    DIA_train_minmax = min_max_scaler_DIA.fit_transform(DIA_TRAIN_tra)
    DIA_test_minmax = min_max_scaler_DIA.transform(DIA_TEST_tra)

    SYS_TRAIN_array = np.array(SYS_BP_train)
    SYS_TRAIN_tra = SYS_TRAIN_array.reshape(-1, 1)
    SYS_TEST_array = np.array(SYS_BP_test)
    SYS_TEST_tra = SYS_TEST_array.reshape(-1, 1)
    min_max_scaler_SYS = preprocessing.MinMaxScaler()
    SYS_train_minmax = min_max_scaler_SYS.fit_transform(SYS_TRAIN_tra)
    SYS_test_minmax = min_max_scaler_SYS.transform(SYS_TEST_tra)


    # 将得到的所有数据形成字典进行保存，之后再永久保存
    train_data={'features':X_train_minmax,'sys_bp':SYS_train_minmax,'dia_bp':DIA_train_minmax}
    test_data ={'features': X_test_minmax, 'sys_bp': SYS_test_minmax, 'dia_bp': DIA_test_minmax}
    # length_train=len(X_train_minmax)
    # length_test=len(X_test_minmax)
    # train_data=[]
    # for ii in range(0,length_train):
    #     train_data.append({'features':X_train_minmax[ii],'sys_bp':SYS_train_minmax[ii],'dia_bp':DIA_train_minmax[ii]})
    #
    # test_data = []
    # for ii in range(0,length_test):
    #     test_data.append(
    #         {'features': X_test_minmax[ii], 'sys_bp': SYS_test_minmax[ii], 'dia_bp': DIA_test_minmax[ii]})

    transform_data=[{'min_max_scaler_X':min_max_scaler_X,'min_max_scaler_DIA':min_max_scaler_DIA,'min_max_scaler_SYS':min_max_scaler_SYS}]

    #     # 将得到的数据保存下来，以备下次使用
    #     print('train_data:%d' % length_train)
    #     train_data_txt=pickle.dumps(train_data)
    #     print('test_data:%d' % length_test)
    #     test_data_txt = pickle.dumps(test_data)
    #     # 归一化的参数也要保存下来，后面的归一化全使用这个
    #     transform_data_txt=pickle.dumps(transform_data)
    #
    #     train_label_data_txt=train_test1_path+'train_label_data_txt'
    #     test_label_data_txt = train_test1_path + 'test_label_data_txt'
    #     transform_data_exist_txt=train_test1_path + 'transform_data_exist_txt'
    #     f=open(train_label_data_txt,'wb')
    #     f.write(train_data_txt)
    #     f.close
    #
    #     f = open(test_label_data_txt, 'wb')
    #     f.write(test_data_txt)
    #     f.close
    #
    #     f = open(transform_data_exist_txt, 'wb')
    #     f.write(transform_data_txt)
    #     f.close
    #
    #
    # else:
    #
    #     f = open(train_test1_path + 'train_label_data_txt', 'rb+')
    #     train_data_txt = f.read()
    #     f.close()
    #     f = open(train_test1_path + 'test_label_data_txt', 'rb+')
    #     test_data_txt = f.read()
    #     f.close()
    #     f = open(train_test1_path + 'transform_data_exist_txt', 'rb+')
    #     transform_data_txt = f.read()
    #     f.close()
    #     train_data = pickle.loads(train_data_txt)
    #     test_data = pickle.loads(test_data_txt)
    #     transform_data=pickle.loads(transform_data_txt)



    return train_data,test_data,transform_data


# def batched_input():
#     # 实现队列的顺序输出
#     pass





def main():

    # 首先构造各种路径和相关参数
    data_path = FLAGS.data_path
    train_test1_path=FLAGS.train_test1_path
    train_rate1 = FLAGS.train_rate1
    train_rate2 = FLAGS.train_rate2
    update = FLAGS.data_update
    # 得到用于模型训练的数据,在其中做了归一化处理
    train_data, test_data,transform_data=input_data(data_path, train_rate1, train_test1_path)
    # 实现队列的输入。。。




if __name__=='__main__':
    main()
