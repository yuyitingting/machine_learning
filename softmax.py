#softmax
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

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
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target

#获取mnist数据
mnist = input_data.read_data_sets("mnist/", one_hot=True)

#变量
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
mean_val=0.0
train_ac=0.0
iteration=7000
batch_size=45000
learning_rate=0.9
test_size=0.18

#损失函数：交叉熵
#cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
#loss=tf.reduce_mean(tf.square(y_-y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

#梯度下降参数优化
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

#划分训练集为训练集：验证集=9：1
x_train, x_validation, y_train, y_validation = train_test_split(mnist.train.images, mnist.train.labels, test_size=test_size)

#正确度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# 用于初始化所有的变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练与验证
for i in range(iteration):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs,batch_ys=next_batcha(x_train,y_train,batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    train_ac+=sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    if(i%20==0):
        mean_val+=sess.run(accuracy, feed_dict={x: x_validation, y_: y_validation})
        #print("validation"+str())+"\n")
print ('W='+str(W)+'\n')
print ('b='+str(b)+'\n')
#测试
a=sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

#记录
with open('./softmax_arg.txt','a',encoding='utf8') as f:
    f.write("\n"+"softmax\niteration="+str(iteration)+"\ncross_entropy\nGradientDescentOptimizer\nlearning_rate="+str(learning_rate)+"\nbatch_size="+str(batch_size)+"\naccuracy= "+str(a)+"\n")
    f.write("mean_validation="+str(mean_val/(iteration/20))+"\n")
    f.write("train_mean_ac="+str(train_ac/(iteration))+'\n')
    f.write("test_size="+str(test_size)+'\n')

#https://www.jianshu.com/p/712519053634