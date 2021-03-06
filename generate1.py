# generating using trained model for net1 & net2
# change to import line to choose a model
# args: <data file> <model file> <lines to generate>

import sys
import pickle
import tensorflow as tf
import numpy as np
from net2 import build_net
from config import *
from util import tictoc

def sample_token(maps, p, cutoff):
    p = p[0, 0]
    p[maps[UNK]] = 0.0
    p[maps[GOS]] = 0.0
    if cutoff:
        p[np.argsort(p)[:-TOPN]] = 0
    p /= np.sum(p)
    return np.random.choice(len(p), p=p)

def gen_seq(sess, input, input_len, state_input, state_output, pred, maps, picks):
    state = sess.run(state_input)
    now = maps[GOS]
    str = ''
    for _ in range(SEQLEN):
        feed = { input: [[now]], input_len: [1], state_input: state }
        p, state = sess.run([ pred, state_output ], feed_dict=feed)
        now = sample_token(maps, p, len(str) > 0)
        if now == maps[EOL]:
            break
        str += picks[now]
    return str

@tictoc('generate sequences')
def generate(src, model, num):
    with open(src, 'rb') as f:
        _, _, picks, maps = pickle.load(f, encoding='binary')

    input, input_len, _, state_input, state_output, _, pred, _ = build_net(len(picks), False)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model)

        for _ in range(num):
            str = gen_seq(sess, input, input_len, state_input, state_output, pred, maps, picks)
            print(str)

if __name__ == '__main__':
    generate(sys.argv[1], sys.argv[2], int(sys.argv[3]))
