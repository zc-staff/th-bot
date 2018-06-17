# attention RNN cell used in net3

import sys
import tensorflow as tf
from tensorflow import nn
from tensorflow import layers
from net1 import test_net

class AttentionCell(nn.rnn_cell.RNNCell):
    def __init__(self, cell, encoder_output, encoder_len, attn_size):
        self._cell = cell
        encoder_size = int(encoder_output.shape[-1])
        self._seq_size = int(encoder_output.shape[-2])

        if type(self._cell.state_size) == tuple:
            self._cell_size = sum(self._cell.state_size)
            self._cell_tuple = True
        else:
            self._cell_size = self._cell.state_size
            self._cell_tuple = False

        self._encoder_len = encoder_len
        self._encoder_output = encoder_output
        with tf.variable_scope(None, 'attn_weight'):
            encoder_weight = tf.get_variable('Ua', shape=(attn_size, encoder_size))
            self._state_weight = tf.get_variable('Wa', shape=(attn_size, self._cell_size))
            self._act_weight = tf.get_variable('va', shape=attn_size)
        with tf.variable_scope('attention'):
            self._encoder_attn = tf.tensordot(encoder_output, encoder_weight, [[-1], [1]])

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state):
        if self._cell_tuple:
            cell_state = tf.concat(state, -1)
        else:
            cell_state = state

        attn = tf.tensordot(cell_state, self._state_weight, [[-1], [1]])
        attn = tf.expand_dims(attn, axis=-2)
        attn = tf.tanh(attn + self._encoder_attn)
        attn = tf.tensordot(attn, self._act_weight, [[-1], [0]])

        mask = tf.sequence_mask(self._encoder_len, self._seq_size, dtype=tf.float32)
        attn = mask * tf.exp(attn)
        attn = attn / tf.reduce_sum(attn, axis=-1, keepdims=True)

        attn = tf.expand_dims(attn, axis=-1)
        attn = tf.reduce_sum(attn * self._encoder_output, axis=-2)
        inputs = tf.concat([ inputs, attn ], -1)
        return self._cell(inputs, state)

def test_attention():
    cell = nn.rnn_cell.GRUCell(512)
    encoder_output = tf.placeholder(tf.float32, shape=(128, 8, 512))
    encoder_len = tf.placeholder(tf.int32, shape=128)
    cell = AttentionCell(cell, encoder_output, encoder_len, 512)
    state_input = cell.zero_state(128, dtype=tf.float32)
    inputs = [ tf.placeholder(tf.float32, shape=(128, 512)) for _ in range(8) ]
    nn.static_rnn(cell, inputs, state_input)

if __name__ == '__main__':
    test_attention()
    test_net(sys.argv[1])
