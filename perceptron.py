import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Threshold Activation Function
def threshold(x):
    cond = tf.less(x, tf.zeros(tf.shape(x), dtype=x.dtype))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


# Plotting Threshold Activation Function
h = np.linspace(-1, 1, 50)
out = threshold(h)

h_sigm = np.linspace(-10, 10, 100)
out_sigm = tf.sigmoid(h_sigm)
out_tanh = tf.tanh(h_sigm)

h_smax = np.linspace(-5, 5, 100)
out_smax = tf.nn.softmax(h_smax)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    y_sigm = sess.run(out_sigm)
    y_tanh = sess.run(out_tanh)
    y_smax = sess.run(out_smax)

plt.xlabel("Activity of Neuron")
plt.ylabel("Output of Neuron")
plt.title("Threshold Activation Function")
plt.plot(h, y)
plt.show()

plt.xlabel("Activity of Neuron")
plt.ylabel("Output of Neuron")
plt.title("Sigmoid Activation Function")
plt.plot(h_sigm, y_sigm)
plt.show()

plt.xlabel("Activity of Neuron")
plt.ylabel("Output of Neuron")
plt.title("Hyperbolic Tangent Activation Function")
plt.plot(h_sigm, y_tanh)
plt.show()

plt.xlabel("Activity of Neuron")
plt.ylabel("Output of Neuron")
plt.title("Softmax Activation Function")
plt.plot(h_smax, y_smax)
plt.show()
