import sys
import pickle
import tensorflow as tf
from util import tictoc
from net3 import build_net

@tictoc('strip model')
def strip(src, model, out):
    with open(src, 'rb') as f:
        _, _, picks, maps = pickle.load(f, encoding='binary')

    encoder_input, encoder_len, encoder_output, decoder_input, state_input, state_output, pred = build_net(len(picks), False)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model)
        saver.save(sess, out)

if __name__ == '__main__':
    strip(sys.argv[1], sys.argv[2], sys.argv[3])
