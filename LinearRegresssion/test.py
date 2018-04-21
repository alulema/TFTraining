import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size = 25
learning_rate = 0.1

sess = tf.Session()
x_data = np.arange(-100, 100)
random_noise = np.random.normal(size=len(x_data)) * 25
y_data = (5 * x_data) + 2 + random_noise

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Raw Data")
plt.plot(x_data, y_data, "b:")
plt.show()


X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

m = tf.Variable(0.0)
b = tf.Variable(0.0)

Y_out = tf.add(tf.multiply(m, X), b)
loss = tf.square(Y_out - Y, name='loss')

init = tf.global_variables_initializer()
sess.run(init)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(optimizer)

loss_vec = []

for i in range(1000):
    rand_index = np.random.choice(len(x_data), size=batch_size)
    rand_x = np.transpose([x_data[rand_index]])
    rand_y = np.transpose([y_data[rand_index]])
    sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})

