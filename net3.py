import sys
import pickle
import tensorflow as tf
from tensorflow import nn
from tensorflow import layers
from config import *
from util import tictoc
from attention import AttentionCell
from net1 import build_loss, build_pred, build_opt, test_net
from net2 import build_embedding, build_preprocess

def build_input(batch_size, encoder_seq, decoder_seq, training):
    encoder_input = tf.placeholder(tf.int32, shape=(batch_size, encoder_seq), name='encoder_input')
    encoder_len = tf.placeholder(tf.int32, shape=(batch_size), name='encoder_len')
    decoder_input = tf.placeholder(tf.int32, shape=(batch_size, decoder_seq), name='decoder_input')
    if training:
        decoder_len = tf.placeholder(tf.int32, shape=(batch_size), name='decoder_len')
        target = tf.placeholder(tf.int32, shape=(batch_size, decoder_seq), name='target')
    else:
        decoder_len = tf.constant(1, dtype=tf.int32, shape=(batch_size,), name='decoder_len')
        target = None
    return encoder_input, encoder_len, decoder_input, decoder_len, target

def build_cell(training):
    cell = [ nn.rnn_cell.GRUCell(LSTMSIZE) for _ in range(LSTMNUM) ]
    if training:
        cell = [ nn.rnn_cell.DropoutWrapper(c, output_keep_prob=DROPOUT) for c in cell ]
    return nn.rnn_cell.MultiRNNCell(cell)

def build_rnn(cell, input, input_len, batch_size, name):
    state_input = cell.zero_state(batch_size, dtype=tf.float32)
    output, state_output = nn.dynamic_rnn(cell, input, input_len, state_input, scope=name)
    return output, state_input, state_output

def build_encoder(input, input_len, embedding, batch_size, training):
    with tf.variable_scope('encoder_pre'):
        input = build_preprocess(input, embedding, training)
    cell = build_cell(training)
    return build_rnn(cell, input, input_len, batch_size, 'encoder')

def build_decoder(input, input_len, embedding, encoder_output, encoder_len, batch_size, training):
    with tf.variable_scope('decoder_pre'):
        input = build_preprocess(input, embedding, training)
    cell = build_cell(training)
    cell = AttentionCell(cell, encoder_output, encoder_len, ATTNSIZE)
    return build_rnn(cell, input, input_len, batch_size, 'decoder')

def build_output(output, vocab_size):
    output = tf.reshape(output, (-1, LSTMSIZE))
    return layers.dense(output, vocab_size)

@tictoc('build net')
def build_net(vocab_size, training):
    batch_size = BATCHSIZE if training else 1
    encoder_seq = SEQLEN
    decoder_seq = SEQLEN if training else 1
    encoder_input, encoder_len, decoder_input, decoder_len, target = build_input(batch_size, encoder_seq, decoder_seq, training)
    embedding = build_embedding(vocab_size)
    encoder_output, _, _ = build_encoder(encoder_input, encoder_len, embedding, batch_size, training)
    decoder_output, state_input, state_output = build_decoder(decoder_input, decoder_len, embedding, encoder_output, encoder_len, batch_size, training)
    with tf.variable_scope('output'):
        decoder_output = build_output(decoder_output, vocab_size)
    if training:
        with tf.variable_scope('target_pre'):
            decoder_target = tf.reshape(tf.one_hot(target, vocab_size), (-1, vocab_size))
        with tf.variable_scope('loss'):
            loss = build_loss(decoder_output, decoder_len, decoder_target, decoder_seq)
        with tf.variable_scope('train'):
            train_opt = build_opt(loss)
        return encoder_input, encoder_len, decoder_input, decoder_len, target, loss, train_opt
    else:
        with tf.variable_scope('pred'):
            pred = build_pred(decoder_output, vocab_size, decoder_seq)
        return encoder_input, encoder_len, encoder_output, decoder_input, state_input, state_output, pred

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        _, _, picks, _ = pickle.load(f, encoding='binary')
    build_net(len(picks), False)
    test_net(sys.argv[2])
