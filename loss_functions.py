import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()
x_function = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

# L2-norm, LSE, Least Squares Error
# ROBUSTNESS: L2-norm squares the error (increasing by a lot if error > 1),
# the model will see a much larger error ( e vs e^2 ) than the L1-norm, so the model
# is much more sensitive to this example, and adjusts the model to minimize this error.
# STABILITY: For any small adjustments of a data point, the regression line will move
# only slightly (regression parameters are continuous functions of the data)
# It is useful when we need to consider any or all outliers
L2_function = tf.square(target - x_function)
L2_output = sess.run(L2_function)

print("-------------------------")
print("LSE - Least Squares Error")
print("-------------------------")
print("Manually: ", sess.run(tf.div(tf.reduce_sum(tf.square(target - x_function)), 2)))
print("TensorFlow: ", sess.run(tf.nn.l2_loss(target - x_function)))


# L1, LAE, Least Absolute Error
L1_function = tf.abs(target - x_function)
L1_output = sess.run(L1_function)

print("--------------------------")
print("LAE - Least Absolute Error")
print("--------------------------")
print("Manually: ", sess.run(tf.reduce_sum(tf.abs(target - x_function))))

delta1 = tf.constant(0.2)
pseudo_huber1 = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_function)/delta1)) - 1.)
pseudo_huber1_output = sess.run(pseudo_huber1)

delta2 = tf.constant(1.)
pseudo_huber2 = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_function) / delta2)) - 1.)
pseudo_huber2_output = sess.run(pseudo_huber2)

delta3 = tf.constant(5.)
pseudo_huber3 = tf.multiply(tf.square(delta3), tf.sqrt(1. + tf.square((target - x_function) / delta2)) - 1.)
pseudo_huber3_output = sess.run(pseudo_huber3)

x_array = sess.run(x_function)
plt.plot(x_array, L2_output, 'b-', label='L2')
plt.plot(x_array, L1_output, 'r--', label='L1')
plt.plot(x_array, pseudo_huber1_output, 'm,', label='Pseudo-Huber (0.2)')
plt.plot(x_array, pseudo_huber2_output, 'k-.', label='Pseudo-Huber (1.0)')
plt.plot(x_array, pseudo_huber3_output, 'g:', label='Pseudo-Huber (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.title('LOSS FUNCTIONS')
plt.show()

x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500, ], 1.)

hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_vals, labels=targets)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=target_dist)
print(sess.run(softmax_xentropy))

unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=sparse_target_dist)
print(sess.run(sparse_xentropy))

x_array = sess.run(x_vals)
plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_out, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, xentropy_weighted_y_out, 'g:', label='Weighted Cross Enropy Loss (x0.5)')
plt.ylim(-1.5, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
