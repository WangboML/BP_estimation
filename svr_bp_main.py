import os
from get_parser import FLAGS

from sklearn.svm import SVR
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from scipy import signal


from data_input import input_data
from data_input_pca import input_data_pca

import matplotlib.pyplot as plt
import numpy as np
import pickle

def train_sys(data_path, data_dir1,train_rate, LR):
    for data in data_dir1:
        train_data, test_data, transform_data = input_data_pca(data_path, data, train_rate)
        standard_scaler_SYS = transform_data['min_max_scaler_SYS']
        X_train=train_data['features']
        Y_train=train_data['sys_bp'].reshape(-1)
        X_test=test_data['features']
        Y_test=test_data['sys_bp'].reshape(-1)
        # Fit regression model


        model_SupportVectorRegressor = SVR(kernel='rbf', C=100, gamma=0.1)
        model_LinearRegression = linear_model.LinearRegression()                                                #线性回归
        model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)                               # 这里使用20个决策树
        model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)


        model1=model_SupportVectorRegressor.fit(X_train,Y_train)
        model2=model_LinearRegression.fit(X_train,Y_train)
        model3 = model_RandomForestRegressor.fit(X_train, Y_train)
        model4 =model_GradientBoostingRegressor.fit(X_train, Y_train)


        Y_pred1=model1.predict(X_test)
        Y_pred2 = model2.predict(X_test)
        Y_pred3 = model3.predict(X_test)
        Y_pred4 = model4.predict(X_test)


        pred_svr_sys = standard_scaler_SYS.inverse_transform(Y_pred1.reshape(-1, 1)).reshape(-1)
        pred_lr_sys = standard_scaler_SYS.inverse_transform(Y_pred2.reshape(-1, 1)).reshape(-1)
        pred_rf_sys = standard_scaler_SYS.inverse_transform(Y_pred3.reshape(-1, 1)).reshape(-1)
        pred_gbrt_sys = standard_scaler_SYS.inverse_transform(Y_pred4.reshape(-1, 1)).reshape(-1)
        real_sys = standard_scaler_SYS.inverse_transform(Y_test.reshape(-1, 1)).reshape(-1)

        # error = np.array(pred_sys).reshape(-1) - np.array(real_sys).reshape(-1)
        #
        # num_5=0
        # num_10=0
        # num_15=0
        # for ii in error:
        #     if abs(ii)<=5:
        #         num_5+=1
        # for ii in error:
        #     if abs(ii)<=10:
        #         num_10+=1
        # for ii in error:
        #     if abs(ii)<=15:
        #         num_15+=1
        #
        # num=len(error)
        # rate_5=num_5/num
        # rate_10=num_10/num
        # rate_15=num_15/num
        #
        # global a,b,c,aami
        # if rate_5>=0.65 and rate_10>=0.85 and rate_15>=0.95:
        #     a+=1
        # elif rate_5>=0.5 and rate_10>=0.75 and rate_15>=0.90:
        #     b+=1
        # elif rate_5 >= 0.4 and rate_10 >= 0.65 and rate_15 >= 0.85:
        #     c+=1
        #
        # mean_val=np.mean(error)
        # std_val=np.std(error)
        # if mean_val<=5 and std_val<=8:
        #     aami+=1

        #下面是画图所需要的东西
        data_sys = './sysdata/'
        filepath = os.path.join(data_sys + 'data12.txt')
        f = open(filepath, 'rb+')
        data_sys = f.read()
        f.close()
        data = pickle.loads(data_sys)

        pred_lstm_sys = np.array(data["pred"]).reshape(-1)

        len_lstm=len(pred_lstm_sys)
        len_data=len(real_sys)
        if len_lstm<len_data:
            pred_svr_sys=pred_svr_sys[::-1]
            pred_lr_sys=pred_lr_sys[::-1]
            pred_rf_sys=pred_rf_sys[::-1]
            pred_gbrt_sys=pred_gbrt_sys[::-1]
            pred_lstm_sys=pred_lstm_sys[::-1]
            real_sys=real_sys[::-1]

        b, a = signal.butter(3, 0.5, 'low')
        real_sys = signal.filtfilt(b, a, real_sys)
        pred_svr_sys = signal.filtfilt(b, a, pred_svr_sys)
        pred_lr_sys = signal.filtfilt(b, a, pred_lr_sys)
        pred_rf_sys = signal.filtfilt(b, a, pred_rf_sys)
        pred_gbrt_sys = signal.filtfilt(b, a, pred_gbrt_sys)
        pred_lstm_sys = signal.filtfilt(b, a, pred_lstm_sys)




        plt.plot(pred_svr_sys, 'g-*', pred_lr_sys, 'b-v', pred_rf_sys, 'c-h', pred_gbrt_sys, 'm-d', pred_lstm_sys,
                     'k-o', real_sys, 'r-D')
        plt.legend(['svr','lr','rf','gbrt','lstm','sys'])
        plt.ylabel(u'收缩压')
        plt.xlabel(u'心搏周期')
        plt.show()


        #real_sys = data["real"]









def train_dia(data_path, data_dir1, train_rate, LR):
    for data in data_dir1:
        train_data, test_data, transform_data = input_data_pca(data_path, data, train_rate)
        standard_scaler_DIA = transform_data['min_max_scaler_DIA']
        X_train=train_data['features']
        Y_train=train_data['dia_bp'].reshape(-1)
        X_test=test_data['features']
        Y_test=test_data['dia_bp'].reshape(-1)
        # Fit regression model
        model_SupportVectorRegressor = SVR(kernel='rbf', C=100, gamma=0.1)
        model_LinearRegression = linear_model.LinearRegression()  # 线性回归
        model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树
        model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)

        model1 = model_SupportVectorRegressor.fit(X_train, Y_train)
        model2 = model_LinearRegression.fit(X_train, Y_train)
        model3 = model_RandomForestRegressor.fit(X_train, Y_train)
        model4 = model_GradientBoostingRegressor.fit(X_train, Y_train)

        Y_pred1 = model1.predict(X_test)
        Y_pred2 = model2.predict(X_test)
        Y_pred3 = model3.predict(X_test)
        Y_pred4 = model4.predict(X_test)

        pred_svr_sys = standard_scaler_DIA.inverse_transform(Y_pred1.reshape(-1, 1)).reshape(-1)
        pred_lr_sys = standard_scaler_DIA.inverse_transform(Y_pred2.reshape(-1, 1)).reshape(-1)
        pred_rf_sys = standard_scaler_DIA.inverse_transform(Y_pred3.reshape(-1, 1)).reshape(-1)
        pred_gbrt_sys = standard_scaler_DIA.inverse_transform(Y_pred4.reshape(-1, 1)).reshape(-1)
        real_sys = standard_scaler_DIA.inverse_transform(Y_test.reshape(-1, 1)).reshape(-1)

        # error=np.array(pred_dia).reshape(-1)-np.array(real_dia).reshape(-1)
        #
        # num_5 = 0
        # num_10 = 0
        # num_15 = 0
        # for ii in error:
        #     if abs(ii) <= 5:
        #         num_5 += 1
        # for ii in error:
        #     if abs(ii) <= 10:
        #         num_10 += 1
        # for ii in error:
        #     if abs(ii) <= 15:
        #         num_15 += 1
        #
        # num = len(error)
        # rate_5 = num_5 / num
        # rate_10 = num_10 / num
        # rate_15 = num_15 / num
        #
        # global a, b, c, aami
        # if rate_5 >= 0.65 and rate_10 >= 0.85 and rate_15 >= 0.95:
        #     a += 1
        # elif rate_5 >= 0.5 and rate_10 >= 0.75 and rate_15 >= 0.90:
        #     b += 1
        # elif rate_5 >= 0.4 and rate_10 >= 0.65 and rate_15 >= 0.85:
        #     c += 1
        #
        # mean_val = np.mean(error)
        # std_val = np.std(error)
        # if mean_val <= 5 and std_val <= 8:
        #     aami += 1

        data_sys = './diadata/'
        filepath = os.path.join(data_sys + 'data12.txt')
        f = open(filepath, 'rb+')
        data_sys = f.read()
        f.close()
        data = pickle.loads(data_sys)

        pred_lstm_sys = np.array(data["pred"]).reshape(-1)

        len_lstm = len(pred_lstm_sys)
        len_data = len(real_sys)
        if len_lstm < len_data:
            pred_svr_sys = pred_svr_sys[::-1]
            pred_lr_sys = pred_lr_sys[::-1]
            pred_rf_sys = pred_rf_sys[::-1]
            pred_gbrt_sys = pred_gbrt_sys[::-1]
            pred_lstm_sys = pred_lstm_sys[::-1]
            real_sys = real_sys[::-1]

        b, a = signal.butter(3, 0.5, 'low')
        real_sys = signal.filtfilt(b, a, real_sys)
        pred_svr_sys = signal.filtfilt(b, a, pred_svr_sys)
        pred_lr_sys = signal.filtfilt(b, a, pred_lr_sys)
        pred_rf_sys = signal.filtfilt(b, a, pred_rf_sys)
        pred_gbrt_sys = signal.filtfilt(b, a, pred_gbrt_sys)
        pred_lstm_sys = signal.filtfilt(b, a, pred_lstm_sys)

        plt.plot(pred_svr_sys, 'g-*', pred_lr_sys, 'b-v', pred_rf_sys, 'c-h', pred_gbrt_sys, 'm-d', pred_lstm_sys,
                 'k-o', real_sys, 'r-D')
        plt.legend(['svr', 'lr', 'rf', 'gbrt', 'lstm', 'sys'])
        plt.ylabel(u'舒张压')
        plt.xlabel(u'心搏周期')
        plt.show()









def main():


    data_path = FLAGS.data_path

    train_rate = FLAGS.train_rate1




    LR = 0.008

    global a,b,c,aami
    a=0
    b=0
    c=0
    aami=0

    data_dir = os.listdir(FLAGS.data_path)

    data_dir1 = ['data1.mat', 'data2', 'data3.mat', 'data4.mat', 'data5.mat', 'data6.mat',
                 'data7.mat', 'data8.mat', 'data9.mat', 'data10.mat', 'data11.mat',
                 'data12.mat', 'data13.mat', 'data14.mat', 'data15.mat', 'data16.mat',
                 'data17.mat', 'data18.mat', 'data19.mat', 'data20.mat', 'realdata1.mat', 'realdata2',
                 'realdata3.mat', 'realdata4.mat', 'realdata5.mat', 'realdata6.mat',
                 'realdata7.mat', 'realdata8.mat', 'realdata9.mat', 'realdata10.mat']
    data_dir1=['data12.mat']
    train_sys(data_path, data_dir1, train_rate,  LR)
    #train_dia(data_path, data_dir1, train_rate,  LR)
    #print(a,b,c,aami)
if __name__=='__main__':
    main()