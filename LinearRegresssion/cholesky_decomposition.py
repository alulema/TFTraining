import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

sess = tf.Session()

x_vals = np.linspace(start=0, stop=10, num=100)
y_vals = x_vals + np.random.normal(loc=0, scale=1, size=100)

x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(a=1, repeats=100)))
X = np.column_stack((x_vals_column, ones_column))
Y = np.transpose(np.matrix(y_vals))
X_tensor = tf.constant(X)
Y_tensor = tf.constant(Y)

tX_X = tf.matmul(tf.transpose(X_tensor), X_tensor)
L = tf.cholesky(tX_X)
tX_Y = tf.matmul(tf.transpose(X_tensor), Y)
sol1 = tf.matrix_solve(L, tX_Y)
sol2 = tf.matrix_solve(tf.transpose(L), sol1)

solution_eval = sess.run(sol2)
m_slope = solution_eval[0][0]
b_intercept = solution_eval[1][0]
print('slope (m): ' + str(m_slope))
print('intercept (b): ' + str(b_intercept))

best_fit = []
for i in x_vals:
    best_fit.append(m_slope * i + b_intercept)

plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Linear Regression', linewidth=3)
plt.legend(loc='upper left')
plt.show()
