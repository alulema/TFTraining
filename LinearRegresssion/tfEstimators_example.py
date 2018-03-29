import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

# 2. Write your function to import data
mnist = input_data.read_data_sets(os.path.join('.', 'mnist'), one_hot=False)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

n_classes = 10
batch_size = 100
n_steps = 1000
learning_rate = 0.01


def model_function(features, labels, mode):

    # 1. Find the most suitable estimator
    # EstimatorSpec fully defines the model to be run by an Estimator.
    espec_op = tf.estimator.EstimatorSpec

    # 3. Define features to be used in data
    # features is a dict as per Estimator specifications
    x = features['images']
    # define the network
    layer_1 = tf.layers.dense(x, 32)
    layer_2 = tf.layers.dense(layer_1, 32)
    logits = tf.layers.dense(layer_2, n_classes)

    # define predicted classes
    predicted_classes = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        espec = espec_op(mode, predictions=predicted_classes)
    else:
        # define loss and optimizer
        entropy_op = tf.nn.sparse_softmax_cross_entropy_with_logits
        loss_op = tf.reduce_mean(entropy_op(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # define accuracy
        accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
        espec = espec_op(mode=mode,
                         predictions=predicted_classes,
                         loss=loss_op,
                         train_op=train_op,
                         eval_metric_ops={'accuracy': accuracy_op})
        return espec


# 4. Instantiate your selected estimator
model = tf.estimator.Estimator(model_function)

# 5. Training!
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_train},
    y=y_train,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True)
model.train(train_input_fn, steps=n_steps)

# 6. Use the trained estimator
# evaluate the model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': x_test},
    y=y_test,
    batch_size=batch_size,
    shuffle=False)
metrics = model.evaluate(eval_input_fn)
print(metrics)
