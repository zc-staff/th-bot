import sys
import pickle
import tensorflow as tf
from tensorflow import nn
from tensorflow import layers
from config import *
from util import tictoc
from net1 import build_loss, build_pred, build_opt, test_net
from net2 import build_embedding
from net3 import build_input, build_encoder, build_output

def build_preprocess(input, embedding, training):
    cell_input = nn.embedding_lookup(embedding, input)
    # if training:
    #     cell_input = nn.dropout(cell_input, DROPOUT)
    return cell_input

def build_cell(training):
    cell = [ nn.rnn_cell.BasicLSTMCell(LSTMSIZE) for _ in range(LSTMNUM) ]
    # if training:
    #     cell = [ nn.rnn_cell.DropoutWrapper(c, output_keep_prob=DROPOUT) for c in cell ]
    return nn.rnn_cell.MultiRNNCell(cell)

def build_encoder(input, input_len, embedding, batch_size, training):
    with tf.variable_scope('encoder_pre'):
        input = build_preprocess(input, embedding, training)
    cell = build_cell(training)
    state_input = cell.zero_state(batch_size, dtype=tf.float32)
    output, state_output = nn.dynamic_rnn(cell, input, input_len, state_input, scope='encoder')
    return output, state_input, state_output

def build_decoder(input, input_len, embedding, encoder_state, batch_size, training):
    with tf.variable_scope('decoder_pre'):
        input = build_preprocess(input, embedding, training)
    cell = build_cell(training)
    if training:
        state_input = encoder_state
    else:
        state_input = cell.zero_state(batch_size, dtype=tf.float32)
    output, state_output = nn.dynamic_rnn(cell, input, input_len, state_input, scope='decoder')
    return output, state_input, state_output

@tictoc('build net')
def build_net(vocab_size, training):
    batch_size = BATCHSIZE if training else 1
    encoder_seq = SEQLEN
    decoder_seq = SEQLEN if training else 1

    encoder_input, encoder_len, decoder_input, decoder_len, target = build_input(batch_size, encoder_seq, decoder_seq, training)
    embedding = build_embedding(vocab_size)

    _, _, encoder_state = build_encoder(encoder_input, encoder_len, embedding, batch_size, training)
    output, state_input, state_output = build_decoder(decoder_input, decoder_len, embedding, encoder_state, batch_size, training)

    with tf.variable_scope('output'):
        output = build_output(output, vocab_size)

    if training:
        with tf.variable_scope('target_pre'):
            decoder_target = tf.reshape(tf.one_hot(target, vocab_size), (-1, vocab_size))
        with tf.variable_scope('loss'):
            loss = build_loss(output, decoder_len, decoder_target, decoder_seq)
        with tf.variable_scope('train'):
            train_opt = build_opt(loss)
        return encoder_input, encoder_len, decoder_input, decoder_len, target, loss, train_opt
    else:
        with tf.variable_scope('pred'):
            pred = build_pred(output, vocab_size, decoder_seq)
        return encoder_input, encoder_len, encoder_state, decoder_input, state_input, state_output, pred

if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        _, _, picks, _ = pickle.load(f, encoding='binary')
    build_net(len(picks), True)
    test_net(sys.argv[2])
