import tensorflow as tf


image_batch = tf.constant([
    [  # First Image
        [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
        [[0, 255, 0], [0, 255, 0], [0, 255, 0]]
    ],
    [  # Second Image
        [[0, 0, 255], [0, 0, 255], [0, 0, 255]],
        [[0, 0, 255], [0, 0, 255], [0, 0, 255]]
    ]
])
image_batch.get_shape()