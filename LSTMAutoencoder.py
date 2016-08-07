
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

import numpy as np

class BasicLSTMAutoencoder(object):
  def __init__(self, hidden_num, inputs, 
    cell=None, optimizer=None, non_reverse=None):
    """Basic version of LSTM-autoencoder.
    (cf. http://arxiv.org/abs/1502.04681)

    Args:
      hidden_num : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors  with size 
              (batch_num x elem_num)
      cell : an rnn cell object (the default option 
            is `BasicLSTMCell` of TensorFlow)
    """

    #self._hidden_num = hidden_num
    #self._inputs = inputs

    if cell is None:
      self._enc_cell = BasicLSTMCell(hidden_num)
      self._dec_cell = BasicLSTMCell(hidden_num)
    else :
      self._enc_cell = cell(hidden_num)
      self._dec_cell = cell(hidden_num)

    with tf.variable_scope('encoder'):
      self.z_codes, enc_state = tf.nn.rnn(
        self._enc_cell, inputs, dtype=tf.float32)

    if non_reverse:
      zero_input = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
      dec_inputs = [zero_input] + inputs[1:]
    else :
      zero_input = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
      dec_inputs = [zero_input] + inputs[-1:0:-1]

    with tf.variable_scope('decoder'):
      dec_output, dec_state = tf.nn.rnn(
        self._dec_cell, dec_inputs, 
        initial_state=enc_state, dtype=tf.float32)

    """the shape of each tensor
      dec_output_ : (step_num x hidden_num)
      dec_weight_ : (hidden_num x elem_num)
      dec_bias_ : (elem_num)
      output_ : (step_num x elem_num)
      input_ : (step_num x elem_num)
    """
    batch_num = inputs[0].get_shape().as_list()[0]
    elem_num = inputs[0].get_shape().as_list()[1]
    with tf.variable_scope('decoder'):
      dec_weight_ = tf.Variable(
        tf.truncated_normal([hidden_num, elem_num], dtype=tf.float32),
        name="dec_weight")
      dec_bias_ = tf.Variable(
        tf.constant(0.1, shape=[elem_num], dtype=tf.float32),
        name="dec_bias")
    dec_output_ = tf.transpose(tf.pack(dec_output), [1,0,2])
    dec_weight_ = tf.tile(tf.expand_dims(dec_weight_, 0), [batch_num,1,1])

    self.output_ = tf.batch_matmul(dec_output_, dec_weight_) + dec_bias_
    self.input_ = tf.transpose(tf.pack(inputs), [1,0,2])
    self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))

    if optimizer is None :
      self.train = tf.train.AdamOptimizer().minimize(self.loss)
    else :
      self.train = optimizer.minimise(self.loss)
