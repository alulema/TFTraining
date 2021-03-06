# import numpy as np
# import tensorflow as tf
#
# x_vals = np.array([1., 3., 5., 7., 9.])
# x_data = tf.placeholder(tf.float32)
# m_const = tf.constant(3.)
# my_product = tf.multiply(x_data, m_const)
# sess = tf.Session()
#
# for x_val in x_vals:
#     print(sess.run(my_product, feed_dict={x_data: x_val}))

# ----------------------------------------------------------

# import tensorflow as tf
# import numpy as np
#
# sess = tf.Session()
# my_array = np.array([[1., 3., 5., 7., 9.],
#                      [-2., 0., 2., 4., 6.],
#                      [-6., -3., 0., 3., 6.]])
# x_vals = np.array([my_array, my_array + 1])
# x_data = tf.placeholder(tf.float32, shape=(3, None))
#
# m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
# m2 = tf.constant([[2.]])
# a1 = tf.constant([[10.]])
#
# prod1 = tf.matmul(x_data, m1)
# prod2 = tf.matmul(prod1, m2)
# add1 = tf.add(prod2, a1)
#
# for x_val in x_vals:
#     print(sess.run(add1, feed_dict={x_data: x_val}))

# ----------------------------------------------------------

import tensorflow as tf
import numpy as np

sess = tf.Session()
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

x_data = tf.placeholder(tf.float32, shape=x_shape)

my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer= tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME''', name='Moving_Avg_Window')


def custom_layer(input_matrix):
    input_matrix_squeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_squeezed)
    temp = tf.add(temp1, b) # Ax + b
    return tf.sigmoid(temp)


with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

print(sess.run(custom_layer1, feed_dict={x_data: x_val}))
