import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

data_path='/home/wangbo/bp/features_data/data/'
data='data2.mat'
datalist_path=os.path.join(data_path,data)
data_list = sio.loadmat(datalist_path)

PPG=np.array(data_list['ppg'].tolist()).reshape(-1)
ECG=np.array(data_list['ecg'].tolist()).reshape(-1)

BP=np.array(data_list['bp'].tolist()).reshape(-1)

plt.plot(PPG)
#plt.ylabel(u'')
plt.xlabel(u'PPG信号')
plt.show()