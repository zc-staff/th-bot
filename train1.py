# training for net1 & net2
# change the import line to choose a model
# args: <data file> <model directory> <max iterations>

import sys
import os
import pickle
import tensorflow as tf
from os.path import join
from net2 import build_net
from data1 import Batches
from config import *
from util import tictoc

@tictoc('save checkpoint')
def after_epoch(idx, sess, saver, summary, model, batches):
    model_path = join(model, str(idx))
    summary_path = join(model, 'summary')

    saver.save(sess, model_path)
    with open(summary_path, 'wb') as fout:
        pickle.dump((summary, batches.state()), fout)

@tictoc('restore checkpoint')
def before_train(sess, saver, model, batches):
    os.makedirs(model, exist_ok=True)
    ckpt = tf.train.get_checkpoint_state(model)
    summary = []
    if ckpt and ckpt.model_checkpoint_path:
        model_path = ckpt.model_checkpoint_path
        summary_path = join(model, 'summary')
        print('restore checkpoint from {}'.format(model))
        saver.restore(sess, model_path)
        with open(summary_path, 'rb') as f:
            summary, state = pickle.load(f, encoding='bytes')
        batches.set_state(state)
    else:
        writer = tf.summary.FileWriter(model, graph=sess.graph)
        writer.close()
        after_epoch(0, sess, saver, summary, model, batches)
    return summary

@tictoc('training')
def train(src, model, iters):
    with open(src, 'rb') as f:
        lines, lines_len, picks, _ = pickle.load(f, encoding='binary')
    batches = Batches(lines, lines_len)
    vocab_size = len(picks)

    input, input_len, target, _, _, loss, _, train_opt = build_net(vocab_size, True)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        summary = before_train(sess, saver, model, batches)

        for i in range(len(summary), iters):
            batch_i, batch_s, batch_t = batches.next_batch()
            feed = { input: batch_i, input_len: batch_s, target: batch_t }
            l, _ = sess.run([ loss, train_opt ], feed_dict=feed)
            print('iteration {}: {}'.format(i, l))
            summary.append(l)

            if (i + 1) % ITERATIONS == 0:
                after_epoch(i + 1, sess, saver, summary, model, batches)

if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]))
