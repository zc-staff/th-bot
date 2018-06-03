import tensorflow as tf
import pickle
import os
from os.path import join
from net1 import build_net
from preprocess2 import data, Batches
from config import *
from util import tictoc

@tictoc('save checkpoint')
def after_epoch(idx, sess, saver, summary, model):
    model_path = join(model, str(idx))
    summary_path = join(model, 'summary')

    saver.save(sess, model_path)
    with open(summary_path, 'wb') as fout:
        pickle.dump(summary, fout)

@tictoc('restore checkpoint')
def before_train(sess, saver, model):
    os.makedirs(model, exist_ok=True)
    ckpt = tf.train.get_checkpoint_state(model)
    summary = []
    if ckpt and ckpt.model_checkpoint_path:
        model_path = ckpt.model_checkpoint_path
        summary_path = join(model, 'summary')
        print('restore checkpoint from {}'.format(model))
        saver.restore(sess, model_path)
        with open(summary_path, 'rb') as f:
            summary = pickle.load(f, encoding='bytes')
    else:
        writer = tf.summary.FileWriter(model, graph=sess.graph)
        writer.close()
        after_epoch(0, sess, saver, summary, model)
    return summary

@tictoc('training')
def train(src, model, iters):
    lines, seq_len, _, _ = data(src)
    batches = Batches(lines, seq_len)

    input, input_len, target, _, _, loss, _, train_opt = build_net(True)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        summary = before_train(sess, saver, model)

        for i in range(len(summary), iters):
            batch_i, batch_s, batch_t = batches.next_batch()
            feed = { input: batch_i, input_len: batch_s, target: batch_t }
            l, _ = sess.run([ loss, train_opt ], feed_dict=feed)
            print('iteration {}: {}'.format(i, l))
            summary.append(l)

            if (i + 1) % ITERATIONS == 0:
                after_epoch(i + 1, sess, saver, summary, model)

if __name__ == '__main__':
    import sys
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]))
