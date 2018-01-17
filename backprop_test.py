import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

x_vals = np.random.normal(loc=1, scale=0.1, size=100)
y_vals = np.repeat(a=10., repeats=100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.multiply(x_data, A)
loss = tf.square(my_output - y_target)

init = tf.initialize_all_variables()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = my_opt.minimize(loss)

print('A starts with value ' + str(sess.run(A)))

for i in range(10000):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1) % 1000 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()

x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

plt.plot(x_vals, y_vals)
plt.show()

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

my_output = tf.add(x_data, A)

my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

init = tf.initialize_all_variables()
sess.run(init)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=my_output_expanded, logits=y_target_expanded)
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(cross_entropy)

for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 200 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(cross_entropy, feed_dict={x_data: rand_x, y_target: rand_y})))
