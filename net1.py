import sys
import tensorflow as tf
from tensorflow import nn
from tensorflow import layers
from config import *
from util import tictoc

def build_input(batch_size, seq_size):
    input = tf.placeholder(tf.int32, shape=(batch_size, seq_size), name='input')
    input_len = tf.placeholder(tf.int32, shape=(batch_size), name='input_len')
    target = tf.placeholder(tf.int32, shape=(batch_size, seq_size), name='target')
    return input, input_len, target

def build_preprocess(input, vocab_size):
    cond = tf.cast(input < vocab_size, tf.int32)
    input = cond * input
    return tf.one_hot(input, vocab_size)

def build_cell(cell_input, input_len, batch_size):
    cell = nn.rnn_cell.MultiRNNCell([ nn.rnn_cell.BasicLSTMCell(LSTMSIZE) for _ in range(LSTMNUM) ])
    state_input = cell.zero_state(batch_size, dtype=tf.float32)
    output, state_output = nn.dynamic_rnn(cell, cell_input, input_len, state_input)
    return output, state_input, state_output

def build_output(output, vocab_size, seq_size):
    output = tf.reshape(output, [ -1, LSTMSIZE ])
    output = layers.dense(output, vocab_size)
    return output

def build_loss(output, input_len, cell_target, seq_size):
    loss = nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=cell_target)
    loss = tf.reshape(loss, [ -1, seq_size ])

    mask = tf.sequence_mask(input_len, seq_size, dtype=tf.float32)
    loss = mask * loss
    loss = tf.reduce_sum(loss, -1)
    loss = tf.reduce_mean(loss)

    return loss

def build_pred(output, vocab_size, seq_size):
    output = nn.softmax(output)
    output = tf.reshape(output, [ -1, seq_size, vocab_size ])
    return output

def build_opt(loss):
    vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, vars), GRADCLIP)
    opt = tf.train.AdamOptimizer(LEARNRATE)
    opt = opt.apply_gradients(zip(grads, vars))
    return opt

@tictoc('build net')
def build_net(train):
    batch_size = BATCHSIZE if train else 1
    seq_size = SEQLEN if train else 1
    input, input_len, target = build_input(batch_size, seq_size)
    cell_input = build_preprocess(input, WORDS)
    output, state_input, state_output = build_cell(cell_input, input_len, batch_size)
    output = build_output(output, WORDS, seq_size)
    cell_target = build_preprocess(target, WORDS)
    loss = build_loss(output, input_len, cell_target, seq_size)
    pred = build_pred(output, WORDS, seq_size)
    train_opt = build_opt(loss)
    return input, input_len, target, state_input, state_output, loss, pred, train_opt

def test_net(output):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(output, graph=sess.graph)
        writer.close()

if __name__ == '__main__':
    build_net(True)
    test_net(sys.argv[1])
