import sys
import os
import pickle
import tensorflow as tf
from time import time
from config import *
from util import tictoc
from net3 import build_net
from train1 import after_epoch, before_train
from data2 import Batches

@tictoc('training')
def train(src, model, iters):
    with open(src, 'rb') as f:
        lines, lines_len, picks, _ = pickle.load(f, encoding='binary')
    batches = Batches(lines, lines_len)
    vocab_size = len(picks)

    encoder_input, encoder_len, decoder_input, decoder_len, target, loss, train_opt = build_net(vocab_size, True)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        summary = before_train(sess, saver, model, batches)

        for i in range(len(summary), iters):
            tic = time()
            ei, el, di, dl, ta = batches.next_batch()
            feed = {
                encoder_input: ei, encoder_len: el,
                decoder_input: di, decoder_len: dl,
                target: ta
            }
            l, _ = sess.run([ loss, train_opt ], feed_dict=feed)
            elapsed = time() - tic
            print('iteration {}: {} in {:.2f}ms'.format(i, l, elapsed * 1000))
            summary.append(l)

            if (i + 1) % ITERATIONS == 0:
                after_epoch(i + 1, sess, saver, summary, model, batches)

if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]))
