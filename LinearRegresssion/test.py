import numpy as np
import tensorflow as tf

sess = tf.Session()
matrix = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.]).reshape(3, 3)
rhs = np.array([1., 2., 3.]).reshape(3, 1)

answer = tf.matrix_solve(matrix, rhs)
print(sess.run(answer))
