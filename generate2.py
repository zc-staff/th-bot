import sys
import pickle
import tensorflow as tf
import numpy as np
from config import *
from util import tictoc
from generate1 import sample_token
from net3 import build_net

@tictoc('generate sequences')
def generate(src, model, num):
    with open(src, 'rb') as f:
        _, _, picks, maps = pickle.load(f, encoding='binary')

    encoder_input, encoder_len, encoder_output, decoder_input, state_input, state_output, pred = build_net(len(picks), False)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model)
        for _ in range(num):
            str = input()
            str = str[:SEQLEN]
            inp = [maps[GOS]]

            def transChar(ch):
                if ch in maps:
                    return maps[ch]
                return maps[UNK]

            inp.extend(list(map(transChar, inp)))
            linp = len(inp)
            inp.extend([ maps[EOL] ] * (SEQLEN - linp))

            eo = sess.run(encoder_output, feed_dict={
                encoder_input: [inp], encoder_len: [linp]
            })

            state = sess.run(state_input)
            now = maps[GOS]
            str = ''
            for _ in range(SEQLEN):
                p, state = sess.run([pred, state_output], feed_dict={
                    encoder_output: eo, encoder_len: [linp],
                    state_input: state, decoder_input: [[now]]
                })
                now = sample_token(maps, p, True)
                if now == maps[EOL]:
                    break
                str += picks[now]
            print(str)

if __name__ == '__main__':
    generate(sys.argv[1], sys.argv[2], int(sys.argv[3]))
