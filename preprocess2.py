import sys
import pickle
import numpy as np
from itertools import islice, tee
from config import *
from util import tictoc

@tictoc('preprocess')
def preprocess(lines):
    pool = ''.join(lines)
    chars = list(set(pool))
    freq = [ (i, pool.count(x)) for i, x in enumerate(chars) ]
    freq.sort(key=lambda x: -x[1])

    picks = [ UNK, GOS, EOL ]
    picks.extend([ chars[x[0]] for x in freq ])
    maps = { x: i for i, x in enumerate(picks) }

    def transChar(x):
        if x in maps:
            return maps[x]
        return maps[UNK]

    def transLine(line):
        l = [maps[GOS]]
        l.extend(list(map(transChar, line)))
        l.extend([ maps[EOL] ] * (SEQLEN + 1 - len(l)))
        return l

    lines = filter(lambda x: len(x) < SEQLEN, lines)
    l1, l2 = tee(lines)

    seq_len = np.array(list(map(lambda x: len(x) + 1, l1)), dtype=np.int32)
    lines = np.array(list(map(transLine, l2)), dtype=np.int32)

    print('{} records'.format(lines.shape[0]))

    return lines, seq_len, picks, maps

def random_samples(num):
    while True:
        for x in np.random.permutation(num):
            yield x

def data(path):
    with open(path, encoding="utf-8") as f:
        lines = [ l.strip() for l in f ]
    return preprocess(lines)

class Batches(object):
    def __init__(self, lines, seq_len):
        self.lines = lines
        self.seq_len = seq_len
        self.perm = np.random.permutation(self.lines.shape[0])
        self.start = 0

    def state(self):
        return self.perm, self.start

    def set_state(self, state):
        self.perm = state[0]
        self.start = state[1]

    def next_batch(self):
        incides = self.perm[self.start:self.start + BATCHSIZE]
        if len(incides) < BATCHSIZE:
            self.perm = np.random.permutation(self.lines.shape[0])
            self.start = BATCHSIZE - len(incides)
            incides = np.append(incides, self.perm[0:self.start])
        else:
            self.start += BATCHSIZE
        input = self.lines[incides, 0:SEQLEN]
        input_len = self.seq_len[incides]
        target = self.lines[incides, 1:SEQLEN + 1]
        return input, input_len, target

if __name__ == '__main__':
    out = data(sys.argv[1])
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(out, f)
