import numpy as np
import tensorflow as tf

class Neural_net:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def add_layer(self, input_to_layer, num_rows, num_cols, activation):
        w_init = np.float32( np.random.rand(num_rows, num_cols) )
        b_init = np.float32( np.random.rand(1, num_cols) )
        weights = tf.get_variable("weights", dtype=tf.float32, initializer=w_init, trainable=True)
        bias = tf.get_variable("bias", dtype=tf.float32, initializer=b_init, trainable=True)
        output = activation( tf.matmul(input_to_layer, weights) + bias )
        return output

    def train_net(self, Y, yHat, epochs, learning_rate):
        with tf.name_scope("loss"):
            loss = tf.reduce_sum(tf.square(Y - yHat))

        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)

        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        writer = tf.summary.FileWriter("D:\deep_net\deepnet2\graph", sess.graph)
        for steps in range(epochs):
            sess.run(train, {X: self.input_data, Y: output_data})
            if steps %1000 == 0: print(sess.run(loss, {X: self.input_data, Y: output_data}))

        print(sess.run(yHat, {X: self.input_data, Y: output_data}))

        writer.close()
        sess.close()


input_data = np.array([[0,0],[1,0],[1,1]], np.float32)
output_data = np.array([[0],[1],[1]], np.float32)

with tf.name_scope("input_sample"):
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

net = Neural_net(input_data, output_data)

with tf.variable_scope("layer1"):
    l1 = net.add_layer(X, 2, 3, tf.nn.sigmoid)
with tf.variable_scope("layer2"):
    l2 = net.add_layer(l1, 3, 1, tf.nn.sigmoid)

net.train_net(Y, l2, 10000, 0.01)
