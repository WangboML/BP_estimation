import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


#global a,b,c,aami
data_sys = './diadata/'
filelist=['data1.txt','data2.txt','data3.txt','data4.txt','data5.txt','data6.txt','data7.txt','data8.txt','data9.txt','data10.txt'
        ,'data11.txt','data12.txt','data13.txt','data14.txt','data15.txt','data16.txt','data17.txt','data18.txt','data19.txt','data20.txt','data21.txt'
    , 'data22.txt','data23.txt','data24.txt','data25.txt','data26.txt','data27.txt','data28.txt','data29.txt','data30.txt']

a = 0
b = 0
c = 0
aami = 0

#for ii in range(len(filelist)):

filepath=os.path.join(data_sys+'data30.txt')
f=open(filepath,'rb+')
data_sys=f.read()
f.close()
data=pickle.loads(data_sys)

pred_sys=data["pred"]

real_sys=data["real"]

error = np.array(pred_sys).reshape(-1) - np.array(real_sys).reshape(-1)

num_5 = 0
num_10 = 0
num_15 = 0
for ii in error:
    if abs(ii) <= 5:
        num_5 += 1
for ii in error:
    if abs(ii) <= 10:
        num_10 += 1
for ii in error:
    if abs(ii) <= 15:
        num_15 += 1

num = len(error)
rate_5 = num_5 / num
rate_10 = num_10 / num
rate_15 = num_15 / num

#global a, b, c, aami
if rate_5 >= 0.65 and rate_10 >= 0.85 and rate_15 >= 0.95:
    a += 1
elif rate_5 >= 0.5 and rate_10 >= 0.75 and rate_15 >= 0.90:
    b += 1
elif rate_5 >= 0.4 and rate_10 >= 0.65 and rate_15 >= 0.85:
    c += 1

mean_val = np.mean(error)
std_val = np.std(error)
if mean_val <= 5 and std_val <= 8:
    aami += 1

print(a,b,c,aami)




