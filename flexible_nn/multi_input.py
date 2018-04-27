import tensorflow as tf

h1 = h2 = h3 = 10
input_dim = 3
output_dim = 3


def generate_nn(s, s2, trainable=True):
    input = tf.concat([s, s2], axis=-1)

    hidden_1 = tf.layers.dense(input, h1, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, name='dense_h1')
    hidden_2 = tf.layers.dense(hidden_1, h2, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, name='dense_h2')

    hidden_3 = tf.layers.dense(hidden_2, h3, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, name='dense_h3')

    actions = tf.layers.dense(hidden_3, output_dim, trainable=trainable)

    return actions


if __name__ == '__main__':
    s_in1 = tf.placeholder(dtype=tf.float32, shape=[None, input_dim - 1])
    s_in2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    srnet = generate_nn(s_in1, s_in2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    concat = tf.concat([s_in1, s_in2], axis=-1)

    print sess.run(concat, feed_dict={s_in1: [[1, 1]], s_in2: [[1]]})
    print sess.run(srnet, feed_dict={s_in1: [[1, 1]], s_in2: [[1]]})
