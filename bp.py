#全连接神经网络
# 加载MNIST数据集
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/", one_hot=True)
from sklearn.model_selection import train_test_split
import numpy

#自定义取随机一个batch内容
def next_batcha(train_data, train_target, batch_size):
    #打乱数据集
    index = [ i for i in range(0,len(train_target)) ]
    numpy.random.shuffle(index);
    #建立batch_data与batch_target的空列表
    batch_data = [];
    batch_target = [];
    #向空列表加入训练集及标签
    for i in range(0,batch_size):
        batch_data.append(train_data[index[i]]);
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target

# 超参数和变量
learning_rate = 0.4
epochs = 10
batch_size = 120
neure=300
stddev=0.03
a=''#记录超参数和变量信息
b=0.0#记录验证集的正确率
d=0.0#记录训练集的正确率
test_size=0.09
# placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 定义参数w和b 全连接层的两个参数w和b都是随机初始化的
W1 = tf.Variable(tf.random_normal([784, neure], stddev=stddev), name='W1')
b1 = tf.Variable(tf.random_normal([neure]), name='b1')

W2 = tf.Variable(tf.random_normal([neure, 10], stddev=stddev), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# 构造隐层网络
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)#激活函数

# 计算预测值
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

#划分训练集为训练集：验证集=9：1
x_train, x_validation, y_train, y_validation = train_test_split(mnist.train.images, mnist.train.labels, test_size=test_size)

# 损失函数
#tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
#loss=tf.reduce_mean(tf.square(y_-y))

# 创建优化器，梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# 初始化
init_op = tf.global_variables_initializer()
# 创建准确率节点，返回一个m乘1的张量，值为T/F
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

a+="\n"+"learning_rate = "+str(learning_rate)+"\n"+"epochs ="+str(epochs)+"\n"+"batch_size ="+str(batch_size)+"\n"+"neure="+str(neure)+"\nstddev="+str(stddev)+"\n"
a+="test_size="+str(test_size)
#a+="\nloss=mean_square"

# 开始训练
with tf.Session() as sess:
    # 变量初始化
    sess.run(init_op)
    # 设定循环100=batch_size使得total_batch次可以循环结束
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = next_batcha(x_train,y_train,batch_size=batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print (sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
        d+=sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        if(epoch%5==0):
            b+=sess.run(accuracy, feed_dict={x:x_validation, y:y_validation})
            print (sess.run(accuracy, feed_dict={x:x_validation, y:y_validation}))
    a+='\ntest_accuracy='+str(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))+"\n"
    a+='validation='+str(b/(epochs/5))+"\n"
    a+='training_ac='+str(d/epochs)

with open('./bp_data.txt','a',encoding='utf8') as f:
    f.write(a)

#https://github.com/zhdzu/MNIST/blob/master/mnist_cnn1.py