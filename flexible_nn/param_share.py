import tensorflow as tf

h1 = h2 = h3 = 16


def generate_sender(obs):
    hidden_1 = tf.layers.dense(obs, h1, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='dense_1')
    hidden_2 = tf.layers.dense(hidden_1, h2, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='dense_2')
    hidden_3 = tf.layers.dense(hidden_2, h3, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='dense_3')

    msg = tf.layers.dense(hidden_3, 2, trainable=True, reuse=tf.AUTO_REUSE, name='dense_4')

    return msg


if __name__ == '__main__':

    obs1 = tf.placeholder(dtype=tf.float32, shape=[None, 9])
    obs2 = tf.placeholder(dtype=tf.float32, shape=[None, 9])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    result1 = generate_sender(obs1)
    result2 = generate_sender(obs2)

    print result1, result2

    cost = tf.reduce_sum(tf.square(y - result1))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    m1 = sess.run(result1, feed_dict={obs1: [[1, 2, 3, 4, 5, 6, 7, 8, 9]]})
    m2 = sess.run(result2, feed_dict={obs2: [[1, 2, 3, 4, 5, 6, 7, 8, 9]]})
    print "Result:", m1, m2

    sess.run(train, feed_dict={obs1: [[1, 2, 3, 4, 5, 6, 7, 8, 9]], y: [[1, 1]]})

    m1 = sess.run(result1, feed_dict={obs1: [[1, 2, 3, 4, 5, 6, 7, 8, 9]]})
    m2 = sess.run(result2, feed_dict={obs2: [[1, 2, 3, 4, 5, 6, 7, 8, 9]]})
    print "Result:", m1, m2



