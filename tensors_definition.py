import tensorflow as tf
import numpy as np

# zeros_tsr = tf.zeros([5, 5], dtype=tf.int32, name='zeros5x5')
# print(zeros_tsr)
# tf.InteractiveSession().run(zeros_tsr)
#
# ones_tsr = tf.ones([5, 5], dtype=tf.float32, name='ones5x5')
# print(ones_tsr)
# tf.InteractiveSession().run(ones_tsr)
#
# filled_tsr = tf.fill([5, 5], 123, name='filled123')
# print(filled_tsr)
# tf.InteractiveSession().run(filled_tsr)
#
# filled2_tsr = tf.constant(123, shape=[5, 5], name='filled123_2', dtype=tf.int16)
# print(filled2_tsr)
# tf.InteractiveSession().run(filled2_tsr)
#
# constant_tsr = tf.constant([1, 2, 3], name='vector')
# print(constant_tsr)
# tf.InteractiveSession().run(constant_tsr)
#
# zeros_similar = tf.zeros_like(constant_tsr)
# print(zeros_similar)
# tf.InteractiveSession().run(zeros_similar)
#
# ones_similar = tf.ones_like(constant_tsr)
# print(ones_similar)
# tf.InteractiveSession().run(ones_similar)
#
# # This tensor defines 7 regular intervals between 0 and 2, 1st param should be float32/64
# linear_tsr = tf.linspace(0., 2, 7)
# print(linear_tsr)
# tf.InteractiveSession().run(linear_tsr)
#
# # This tensor defines 4 elements between 6 and 17, with a delta of 3
# int_seq_tsr = tf.range(start=6, limit=17, delta=3)
# print(int_seq_tsr)
# tf.InteractiveSession().run(int_seq_tsr)

# # Random numbers from uniform distribution
# rand_unif_tsr = tf.random_uniform([5, 5], minval=0, maxval=1)
# print(rand_unif_tsr)
# tf.InteractiveSession().run(rand_unif_tsr)
#
# # Random numbers from normal distribution
# rand_normal_tsr = tf.random_normal([5, 5], mean=0.0, stddev=1.0)
# print(rand_normal_tsr)
# tf.InteractiveSession().run(rand_normal_tsr)
#
# # Random numbers from normal distribution, limitating values within 2 SD from mean
# trunc_norm_tsr = tf.truncated_normal([5, 5], mean=0.0, stddev=1.0)
# print(trunc_norm_tsr)
# tf.InteractiveSession().run(trunc_norm_tsr)
#
# # Shuffles existing tensor
# seq = tf.linspace(0., 7, 8)
# tf.InteractiveSession().run(seq)
# tf.InteractiveSession().run(tf.random_shuffle(seq))
#
# # Random crop on existing tensor to specified dimension
# tf.InteractiveSession().run(tf.random_crop(seq, [3, ]))
#
# np_array = np.array([[1, 2], [3, 4]])
# np_tsr = tf.convert_to_tensor(np_array, dtype=tf.int32)
# print(np_tsr)
# tf.InteractiveSession().run(np_tsr)
#
# seq_var = tf.Variable(seq)
#
# # Initialize variables in session
# sess = tf.Session()
# initialize_op = tf.global_variables_initializer()
# sess.run(initialize_op)
#
# sess = tf.Session()
# x = tf.placeholder(tf.float32, shape=[2, 2])
# # y is the operation to run on x placeholder
# y = tf.identity(x)
# # x_vals is data to feed into the x placeholder
# x_vals = np.random.rand(2, 2)
# # Runs y operation
# sess.run(y, feed_dict={x: x_vals})

sess = tf.Session()
first_var = tf.Variable(tf.lin_space(0., 7, 8), name='1st_var')
sess.run(first_var.initializer)
second_var = tf.Variable(tf.zeros_like(first_var), name='2nd_var')
# Depends on first_var
print(sess.run(second_var.initializer))
