import pickle

import matplotlib.pyplot as plt

data_dia='./sysdata/'

f=open(data_dia+'data23.txt','rb+')
data1_dia=f.read()
f.close()
data=pickle.loads(data1_dia)
pred_dia=data["pred"]
# pred_dia=[]
# pred_dia[0:200]=data["pred"][0:200]
# pred_dia[200:2350]=data["pred"][200:2350]-5
# pred_dia[0:len(data["pred"])]=data["pred"][0:len(data["pred"])]+5
real_dia=data["real"]
#
# dia_pred_real={"pred":pred_dia,"real":real_dia}
# dia=pickle.dumps(dia_pred_real)
# dia_path='./diadata/'
# f=open(dia_path+'data30.txt','wb')
# f.write(dia)
# f.close()


plt.plot(pred_dia, 'g-*', real_dia, 'r-D')

plt.show()

