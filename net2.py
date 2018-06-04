import sys
import pickle
import tensorflow as tf
from tensorflow import nn
from tensorflow import layers
from config import *
from util import tictoc
from net1 import build_input, build_loss, build_pred, build_opt, test_net

def build_embedding(vocab_size):
    init = tf.random_uniform_initializer(-1, 1)
    return tf.get_variable('embedding', (vocab_size, LSTMSIZE), initializer=init)

def build_preprocess(input, embedding, training):
    cell_input = nn.embedding_lookup(embedding, input)
    if training:
        cell_input = nn.dropout(cell_input, DROPOUT)
    return cell_input

def build_cell(cell_input, input_len, batch_size, training):
    cell = [ nn.rnn_cell.BasicLSTMCell(LSTMSIZE) for _ in range(LSTMNUM) ]
    if training:
        cell = [ nn.rnn_cell.DropoutWrapper(c, output_keep_prob=DROPOUT) for c in cell ]
    cell = nn.rnn_cell.MultiRNNCell(cell)
    state_input = cell.zero_state(batch_size, dtype=tf.float32)
    output, state_output = nn.dynamic_rnn(cell, cell_input, input_len, state_input)
    return output, state_input, state_output

def build_output(output, embedding):
    output = tf.reshape(output, (-1, LSTMSIZE))
    embedding = tf.transpose(embedding)
    o_sum = tf.reduce_sum(output * output, axis=1, keepdims=True)
    e_sum = tf.reduce_sum(embedding * embedding, axis=0, keepdims=True)
    crs = tf.matmul(output, embedding)
    return crs - 0.5 * (o_sum + e_sum)

@tictoc('build net')
def build_net(vocab_size, training):
    batch_size = BATCHSIZE if training else 1
    seq_size = SEQLEN if training else 1
    input, input_len, target = build_input(batch_size, seq_size)
    embedding = build_embedding(vocab_size)
    cell_input = build_preprocess(input, embedding, training)
    output, state_input, state_output = build_cell(cell_input, input_len, batch_size, training)
    output = build_output(output, embedding)
    cell_target = tf.reshape(tf.one_hot(target, vocab_size), (-1, vocab_size))
    loss = build_loss(output, input_len, cell_target, seq_size)
    pred = build_pred(output, vocab_size, seq_size)
    train_opt = build_opt(loss)
    return input, input_len, target, state_input, state_output, loss, pred, train_opt

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        _, _, picks, _ = pickle.load(f, encoding='binary')
    build_net(len(picks), True)
    test_net(sys.argv[2])
