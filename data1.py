import numpy as np
from config import *

class Batches(object):
    def __init__(self, lines, seq_len):
        self.lines = lines
        self.seq_len = seq_len
        self.size = self.lines.shape[0]
        self.perm = np.random.permutation(self.size)
        self.start = 0

    def state(self):
        return self.perm, self.start

    def set_state(self, state):
        self.perm = state[0]
        self.start = state[1]

    def next_incides(self):
        incides = self.perm[self.start:self.start + BATCHSIZE]
        if len(incides) < BATCHSIZE:
            self.perm = np.random.permutation(self.size)
            self.start = BATCHSIZE - len(incides)
            incides = np.append(incides, self.perm[0:self.start])
        else:
            self.start += BATCHSIZE
        return incides

    def next_batch(self):
        incides = self.next_incides()
        input = self.lines[incides, 0:SEQLEN]
        input_len = self.seq_len[incides]
        target = self.lines[incides, 1:SEQLEN + 1]
        return input, input_len, target
