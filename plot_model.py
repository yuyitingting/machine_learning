import matplotlib.pyplot as plt
from pylab import *  # 支持中文

mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['5', '10', '15', '20', '25']
#softmax
x = [10000,15000,20000,25000,30000,35000,40000]
y = [0.9914,0.9921,0.9935,0.9921,0.9949,0.9914,0.9928]
y1=[0.9362,0.9394,0.9410,0.9433,0.9471,0.9479,0.9404]
#neural_net
# x=[10000,15000,20000,25000,30000,35000,40000,45000]
# y=[1.0, 0.9990, 0.9980, 0.9980, 0.9975, 0.9985, 0.9975, 0.9990]
# y1=[0.9653, 0.9659, 0.9675, 0.9678, 0.9618, 0.9618, 0.9593, 0.9545]
plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'Training accuracy')
plt.plot(x, y1, marker='o', mec='r', mfc='w',label=u'Validation accuracy')
plt.legend() # 让图例生效
plt.xlabel(u"number of training samples") #X轴标签
plt.ylabel("accuracy") #Y轴标签
plt.title("bp learning curve") #标题
# plt.title("Neural Network learning curve") #标题
plt.show()