
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

import numpy as np

class BasicLSTMAutoencoder(object):
  def __init__(self, num_units, inputs, cell=None, optimizer=None):
    """ 
    Args:
      num_units : number of hidden elements of each LSTM unit.
      inputs : a list of input tensor
      cell : an rnn cell object (the default option 
            is `BasicLSTMCell` of TensorFlow)
    """

    self._num_units = num_units
    self._inputs = inputs

    if cell is None:
      self._cell = BasicLSTMCell(num_units)
    else :
      self._cell = cell

    self.z_codes, enc_state = tf.nn.rnn(
      self._cell, inputs, dtype=tf.float32)

    zero_input = tf.zeros(tf.shape(inputs), dtype=tf.float32)
    dec_inputs = [zero_input] + inputs[-1:0:-1]

    dec_output, dec_state = tf.nn.rnn(
      dec_inputs, initial_state=enc_state, dtype=tf.float32)

    input_t = tf.pack(inputs)
    dec_output_t = tf.pack(inputs)
    self.loss = reduce_mean(tf.square(input_t - dec_output_t))

    if optimizer is None :
      self.train = tf.train.AdamOptimizer().minimize(self.loss)
    else :
      self.train = optimizer.minimise(self.loss)
