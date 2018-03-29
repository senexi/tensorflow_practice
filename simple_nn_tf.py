import numpy as np  
import pandas as pd  
import tensorflow as tf

learning_rate = 0.5
epochs = 10

# The training set. We have 4 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = np.array([0, 1, 1, 0]).T

# input playeholder
x = tf.placeholder(tf.float32, [None, 3])
# output placeholder
y = tf.placeholder(tf.float32, [None, 1])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([3, 5], stddev=0.03), name='weights_input_hidden_layer')
b1 = tf.Variable(tf.random_normal([5]), name='bias_hidden_layer')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([5, 1], stddev=0.03), name='weights_hidden_output_layer')
b2 = tf.Variable(tf.random_normal([1]), name='bias_output_layer')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
# apply the activation function (sigmoid or relu e.g.)
hidden_out = tf.nn.relu(hidden_out)

# calculate the output of the output layer
output_out = tf.add(tf.matmul(hidden_out, W2), b2)
# apply the activation function (sigmoid, relu or softmax e.g.)
y_ = tf.nn.softmax(output_out)


# calculate the costs
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cost = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   for epoch in range(epochs):
        avg_cost = 0
        _, c = sess.run([optimiser, cost], feed_dict={x: training_set_inputs, y: training_set_outputs})
        avg_cost += c / epochs
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: np.array([1, 0, 0]), y: np.array([1])}))