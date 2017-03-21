from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = {'directory': 'tmp/mnist', 'validation_size':5000}

def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

def csv_input():

  COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
  FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
  LABEL = "medv"

  training_set = pd.read_csv("data/boston_train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)

  test_set = pd.read_csv("data/boston_test.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

  prediction_set = pd.read_csv("data/boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  feature_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in FEATURES]

  regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[10, 10],
                                            model_dir="tmp/boston_model")

  regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

  #evaluation
  ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS['directory'], name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

def record_input():
  data_sets = mnist.read_data_sets(FLAGS['directory'],
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS['validation_size'])
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')

record_input()