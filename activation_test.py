import tensorflow as tf

sess = tf.Session()
# x = tf.lin_space(-3., 3., 24)
# print(sess.run(tf.nn.sigmoid(x)))

x = tf.lin_space(-3., 9., 24)
print(sess.run(tf.nn.relu6(x)))

x = tf.lin_space(-5., 5., 24)
print(sess.run(tf.nn.tanh(x)))


