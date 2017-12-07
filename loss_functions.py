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
#
L1_y_vals = tf.abs(target - x_function)
L1_y_out = sess.run(L1_y_vals)

print("--------------------------")
print("LAE - Least Absolute Error")
print("--------------------------")
print("Manually: ", sess.run(tf.reduce_sum(tf.abs(target - x_function))))
