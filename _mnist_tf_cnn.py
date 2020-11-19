from tensorflow.examples.tutorials.mnist import input_data
# mnist.train：訓練用資料 55,000 筆。
# mnist.validation：驗證用資料 5,000 筆。
# mnist.test：測試用資料 10,000 筆。
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


Learning_Rate       = 1e-4
Num_Iterations      = 1000
Batch_Size          = 10
Num_Input           = 784
Num_Classes         = 10


x   = tf.placeholder(tf.float32, [None, Num_Input]) # Input Data, Num_Input = 784 = 28*28 ( image size ), None = batch size.
y_  = tf.placeholder(tf.float32, [None, Num_Classes]) # Label "數字 0 ~ 9"

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32]) # conv1 Setting.
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # Conv1 with ReLU.
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64]) # conv2 Setting.
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # Conv2 with ReLU.
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([7 * 7 * 64, 1024]) # FC Setting.
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # Flatten Layer.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # FC Layer 1 with ReLU.

keep_prob   = tf.placeholder(tf.float32) # Dropout Setting.
h_fc1_drop  = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # Ouput Data from FC Layer 2.

# prediction = tf.nn.softmax(y_out) # Probability of every category.
# ans = tf.argmax(prediction, 1)

# Training
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out)) # Loss Function.
train_step          = tf.train.AdamOptimizer(Learning_Rate).minimize(cross_entropy) # Optimizer.

# Evaluation
correct_prediction  = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 初始化 tf.Variable.
    for i in range(Num_Iterations):
        batch = mnist.train.next_batch(Batch_Size) # Batch Size.

        # Calculate accuracy every 100 iterations.
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0}) # accuracy.eval() = accuracy.run(). 前者可以回傳參數，後者無法.
            print('step %d, training accuracy %g' % (i, train_accuracy) )
        # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5} )
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5} ) # 同上
    
    # Use plt or cv imshow
    plt.imshow(mnist.test.images[0].reshape(28, 28) )
    plt.show()
    # cv.imshow("test", mnist.test.images[0].reshape(28, 28) )
    # cv.waitKey(0)

    # If OOM
    test_num = 0 # 紀錄 test 迴圈次數
    test_accuracy = 0 # 累積的準確率
    for j in range(mnist.test.num_examples//Batch_Size): # 每次取 100 個計算準確率, mnist.test.num_examples = test 樣本數量
        test_accuracy += accuracy.eval(feed_dict={x: mnist.test.images[j: j+99], y_: mnist.test.labels[j: j+99], keep_prob: 1.0})
        j += 100
        test_num += 1
    test_accuracy /= test_num
    print('test accuracy %g' % test_accuracy)

    # Else
    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
