import tensorflow as tf

a = tf.constant([5, 3], name="input_a")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b,c, name="add_d")

sess = tf.Session()
print(sess.run(d))

# visualize by py -3 -m tensorflow.tensorboard logdir="my_graph"
writer = tf.summary.FileWriter('./my_graph', sess.graph)

writer.close()
sess.close()
