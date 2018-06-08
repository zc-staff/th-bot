import sys
import pickle
import tensorflow as tf
import numpy as np
from config import *
from util import tictoc
from generate1 import sample_token
from net4 import build_net
from data2 import Batches

@tictoc('generate sequences')
def generate(src, model, num):
    with open(src, 'rb') as f:
        lines, lines_len, picks, maps = pickle.load(f, encoding='binary')

    # batches = Batches(lines, lines_len)
    # ei, el, di, dl, ta = batches.next_batch()
    # str = ''
    # for x in ei[0]:
    #     if x == maps[EOL]:
    #         break
    #     str += picks[x]
    # print(str)

    # encoder_input, encoder_len, decoder_input, decoder_len, target, loss, train_opt = build_net(len(picks), True)
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     saver.restore(sess, model)
    #     l = sess.run(loss, feed_dict={
    #         encoder_input: ei, encoder_len: el,
    #         decoder_input: di, decoder_len: dl,
    #         target: ta
    #     })
    #     print(l)

    encoder_input, encoder_len, encoder_state, decoder_input, state_input, state_output, pred = build_net(len(picks), False)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model)
        for _ in range(num):
            str = input()
            str = str[:SEQLEN]

            def transChar(ch):
                if ch in maps:
                    return maps[ch]
                return maps[UNK]

            inp = list(map(transChar, str))
            linp = len(inp)
            inp.extend([ maps[EOL] ] * (SEQLEN - linp))
            # print(inp)
            # print(linp)

            state = sess.run(encoder_state, feed_dict={
                encoder_input: [inp], encoder_len: [linp]
            })

            now = maps[GOS]
            str = ''
            for _ in range(SEQLEN):
                p, state = sess.run([pred, state_output], feed_dict={
                    state_input: state, decoder_input: [[now]]
                })
                now = sample_token(maps, p, True)
                if now == maps[EOL]:
                    break
                str += picks[now]
            print(str)

if __name__ == '__main__':
    generate(sys.argv[1], sys.argv[2], int(sys.argv[3]))
