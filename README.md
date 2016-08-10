# LSTM-autoencoder
TensorFlow LSTM-autoencoder implementation

## Usage
```python

# hidden_num : the number of hidden units in each RNN-cell
# inputs : a list of tensor with size (batch_num x elem_num)
ae = LSTMAutoencoder(hidden_num, inputs)
...
sess.run(init)
...
sess.run(ae.train, feed_dict={input_variable:input_array,...})
```

## Reference
Unsupervised Learning of Video Representations using LSTMs (Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov)

http://arxiv.org/abs/1502.04681
