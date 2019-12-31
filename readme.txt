softmax分类器
    代码：softmax.py
    实验结果：softmax_data.txt softmax_data2.txt softmax_plot.txt
全连接神经网络
    代码：bp.py
    实验结果：bp_data.txt
画learning curve代码：plot_model.py


实验步骤
一．	Softmax
获取数据
将训练数据划分为训练集和验证集
预测函数：tf.nn.softmax(tf.matmul(x, W) + b)
选取损失函数：交叉熵或均方差
选取梯度下降法进行参数优化
在指定迭代次数下进行训练，每次迭代以batch size为单位训练全部训练集的数据
其中batch size使用自定义函数获取，随机在训练集获取指定大小的训练样本
计算训练集、验证集、测试集上的平均识别准确率

二．	Fully-connected neural network
获取数据
将训练数据划分为训练集和验证集
定义一个三层的全连接神经网络
输入层和隐藏层间使用relu激活函数
隐藏层和输出层间使用softmax函数
预测函数：hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
选取损失函数：交叉熵损失函数或均方差
选取更新策略：固定学习率，阶梯型指数衰减，标准指数衰减
选取梯度下降进行参数优化
在指定迭代次数下进行训练，每次迭代以batch size为单位训练全部训练集的数据
其中batch size使用自定义函数获取，随机在训练集获取指定大小的训练样本
计算训练集、验证集、测试集上的平均识别准确率


