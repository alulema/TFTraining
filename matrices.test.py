import tensorflow as tf
import numpy as np

sess = tf.Session()

identity_matrix = tf.diag([1., 1., 1., 1., 1.])
mat_A = tf.truncated_normal([5, 2], dtype=tf.float32)
mat_B = tf.constant([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]])
mat_C = tf.random_normal([5, ], mean=0, stddev=1.0)
mat_D = tf.convert_to_tensor(np.array([[1.2, 2.3, 3.4], [4.5, 5.6, 6.7], [7.8, 8.9, 9.10]]))

# Matrix Operations
print("A + B:")
print(sess.run(mat_A + mat_B))

print("B - B:")
print(sess.run(mat_B - mat_B))

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
c = tf.matmul(a, b)
print("B * Identity")
C = tf.transpose(mat_B)
print(sess.run(tf.matmul(C, identity_matrix)))

print("Transposed C")
print(sess.run(tf.transpose(mat_C)))

print("Matrix Determinant D")
print(sess.run(tf.matrix_determinant(mat_D)))

print("Matrix Inverse D")
print(sess.run(tf.matrix_inverse(mat_D)))

print("Cholesky decomposition")
print(sess.run(tf.cholesky(identity_matrix)))

print("Eigen decomposition")
print(sess.run(tf.self_adjoint_eig(mat_D)))

# Element-wise operations
print("A + B (Element-wise):")
print(sess.run(tf.add(mat_A, mat_B)))

print("A - B (Element-wise):")
print(sess.run(tf.subtract(mat_A, mat_B)))

print("A * B (Element-wise):")
print(sess.run(tf.multiply(mat_A, mat_B)))

print("A % B (Element-wise):")
print(sess.run(tf.div([2, 2], [5, 4])))
print("A / B (Element-wise):")
print(sess.run(tf.truediv([2, 2], [5, 4])))
print("A / B Floor-approximation (Element-wise):")
print(sess.run(tf.floordiv([8, 8], [5, 4])))
print(sess.run(tf.floor(tf.truediv([8, 8], [5, 4]))))
print("A/B Remainder (Element-wise):")
print(sess.run(tf.mod([8, 8], [5, 4])))

# Cross-product
print("Cross-product:")
print(sess.run(tf.cross([1, -1, 2], [5, 1, 3])))

print("A:")
print(sess.run(mat_A))
print("abs(A):")
print(sess.run(tf.abs(mat_A)))
print("ceil(A):")
print(sess.run(tf.ceil(mat_A)))
print("exp(A):")
print(sess.run(tf.exp(mat_A)))
print("maximum(A):")
print(sess.run(tf.maximum(mat_A, mat_B)))
print("minimum(A):")
print(sess.run(tf.minimum(mat_A, mat_B)))
print("pow(A):")
print(sess.run(tf.pow(mat_A, mat_B)))

print("Tangent function(tan(pi/4)=1):")
print(sess.run(tf.tan(tf.truediv(np.pi, 4.))))
print("Tangent function(tan(pi/4)=sin(pi/4)/cos(pi/4)=1):")
print(sess.run(tf.div(tf.sin(np.pi / 4.), tf.cos(np.pi / 4.))))


# y=3x^2-x+10
def custom_polynomial(value):
    return tf.subtract(3 * tf.square(value), value) + 10


print("Custom_polynomial:")
print(sess.run(custom_polynomial(11)))
