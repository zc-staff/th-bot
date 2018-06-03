import tensorflow as tf
from tensorflow import nn
from tensorflow import layers
from config import *
from util import tictoc

def build_input(batch_size, seq_size):
    input = tf.placeholder(tf.int32, shape=(batch_size, seq_size))
    input_len = tf.placeholder(tf.int32, shape=(batch_size))
    target = tf.placeholder(tf.int32, shape=(batch_size, seq_size))
    return input, input_len, target

def build_cell(input, input_len, batch_size):
    input = tf.one_hot(input, WORDSALL)
    cell = nn.rnn_cell.MultiRNNCell([ nn.rnn_cell.BasicLSTMCell(LSTMSIZE) for _ in range(LSTMNUM) ])
    state_input = cell.zero_state(batch_size, dtype=tf.float32)
    output, state_output = nn.dynamic_rnn(cell, input, input_len, state_input)
    return output, state_input, state_output

def build_output(output, seq_size):
    output = tf.reshape(output, [ -1, LSTMSIZE ])
    output = layers.dense(output, WORDSALL)
    output = tf.reshape(output, [ -1, seq_size, WORDSALL ])
    return output

def build_loss(output, input_len, target, batch_size, seq_size):
    target = tf.one_hot(target, WORDSALL)
    loss = nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=target)

    mask = tf.expand_dims(tf.range(seq_size), 0) < tf.expand_dims(input_len, -1)
    mask = tf.cast(mask, tf.float32)
    loss = mask * loss
    loss = tf.reduce_sum(loss, -1)
    loss = tf.reduce_mean(loss)

    return loss

def build_pred(output):
    output = nn.softmax(output)
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
    output, state_input, state_output = build_cell(input, input_len, batch_size)
    output = build_output(output, seq_size)
    loss = build_loss(output, input_len, target, batch_size, seq_size)
    pred = build_pred(output)
    train_opt = build_opt(loss)
    return input, input_len, target, state_input, state_output, loss, pred, train_opt

if __name__ == '__main__':
    import sys
    build_net(True)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(sys.argv[1], graph=sess.graph)
        writer.close()
