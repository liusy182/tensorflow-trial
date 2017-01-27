import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.mul(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

sess = tf.Session()
print(sess.run(e))

# visualize by py -3 -m tensorflow.tensorboard logdir="my_graph"
writer = tf.summary.FileWriter('./tmp/my_graph', sess.graph)

writer.close()
sess.close()
