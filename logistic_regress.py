import tensorflow as tf
import os

# initialize variables/model parameters
W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")


# former inference is now used for combining inputs
def combine_inputs(X):
    return tf.matmul(X, W) + b

# new inferred value is the sigmoid applied to the former
def inference(X):
    return tf.sigmoid(combine_inputs(X))

def loss(X, Y):
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

def inputs():
    weight_age = [
        [84, 46], [73, 20], [65, 52], [70, 30], [76, 57], 
        [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], 
        [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], 
        [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], 
        [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]
    ]
    blood_fat_content = [
        354, 190, 405, 263, 451, 
        302, 288, 385, 402, 365, 
        209, 290, 346, 254, 395, 
        434, 220, 374, 308, 220, 
        311, 181, 274, 303, 244
    ]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)

def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    print(sess.run(inference([[80., 25.]])))
    print(sess.run(inference([[65., 25.]])))


saver = tf.train.Saver()
# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    initial_step = 0

    # verify if we don't have a checkpoint saved already
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    # actual training loop
    training_steps = 1000
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ")
            print(sess.run([total_loss]))
        if step % 1000 == 0:
            saver.save(sess, 'tmp/my-model', global_step=step)

    saver.save(sess, 'tmp/my-model', global_step=training_steps)

    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()