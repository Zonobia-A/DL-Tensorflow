import tensorflow as tf
import numpy as np

# parameters
learning_rate = 0.01
epochs = 10000
batch_size = 5

# data
x_data = np.random.rand(10)
y_data = np.random.rand(10) + 100 * x_data + 10

# graph
X = tf.placeholder(dtype=tf.float32, shape=[None], name="x")
Y = tf.placeholder(dtype=tf.float32, shape=[None], name="y")
W = tf.Variable(tf.constant(1.0, dtype=tf.float32), name="w")
b = tf.Variable(tf.constant(0.0, dtype=tf.float32), name="b")
z = X * W + b
loss = tf.reduce_mean(tf.square(z - Y))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()

# train
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		total_batch = x_data.shape[0] // batch_size
		for batch in range(total_batch):
			x = x_data[batch * batch_size : (batch + 1) * batch_size]
			y = y_data[batch * batch_size : (batch + 1) * batch_size]
			sess.run(optimizer, feed_dict={X: x, Y: y})
		print("epoch:", epoch, "W:", sess.run(W, feed_dict={X: x, Y: y}), "b:", sess.run(b, feed_dict={X: x, Y: y}))

# 心得：写placeholder的时候注意None的用法，还有注意tensorflow的矩阵尺寸的问题