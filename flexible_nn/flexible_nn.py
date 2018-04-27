import tensorflow as tf


def run():
    x = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 2])
    z = tf.placeholder(tf.float32, [None, 2])
    check = tf.placeholder(tf.bool, name='check')

    x_data = [[1, 1], [2, 2], [3, 3]]
    y_data = [[1, 1], [2, 2], [3, 3]]
    z_data = [[1, 1], [2, 2], [3, 3]]

    zero = tf.zeros([1, 2], dtype=tf.float32, name=None)

    output = tf.cond(check[0], lambda: tf.add(x, zero), lambda: zero)
    output = tf.cond(check[1], lambda: tf.add(y, output), lambda: output)
    output = tf.cond(check[2], lambda: tf.add(z, output), lambda: output)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={check: [False, True, True], x: [[3, 1]], y: [[3, 1]], z: [[3, 1]]}))
        print(sess.run(output, feed_dict={check: [True, True, True], x: [[3, 1]], y: [[3, 1]], z: [[3, 1]]}))
        print(sess.run(output, feed_dict={check: [False, True, True], x: x_data, y: y_data, z: z_data}))


if __name__ == '__main__':
    run()
