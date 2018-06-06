import numpy as np
import pickle
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from os.path import join
from util import tictoc

@tictoc('smoothing')
def smooth(x, wlen):
    s = np.r_[x[wlen - 1:0:-1], x, x[-1:-wlen:-1]]
    w = np.hamming(wlen)
    y = np.convolve(w / np.sum(w), s, mode='valid')
    return y[(wlen // 2 - 1):-(wlen // 2)]

@tictoc('plot summary')
def plot(model, out):
    summary_path = join(model, 'summary')
    with open(summary_path, 'rb') as f:
        summary, _ = pickle.load(f, encoding='bytes')
    summary = smooth(np.array(summary), 32)

    plt.figure(1)
    plt.plot(summary, label='loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.grid(True)
    plt.legend()
    plt.savefig(out)

if __name__ == '__main__':
    import sys
    plot(sys.argv[1], sys.argv[2])
