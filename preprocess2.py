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

    picks = [ chars[x[0]] for x in freq[:WORDS] ]
    picks.append(GOS)
    picks.append(EOL)
    picks.append(UNK)
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
        self.samples = random_samples(lines.shape[0])

    def next_batch(self):
        incides = list(islice(self.samples, BATCHSIZE))
        input = self.lines[incides, 0:SEQLEN]
        input_len = self.seq_len[incides]
        target = self.lines[incides, 1:SEQLEN + 1]
        return input, input_len, target

if __name__ == '__main__':
    import sys
    l, s, p, m = data(sys.argv[1])
    print(l.shape)
    print(s)
    batches = Batches(l, s)
    input, input_len, target = batches.next_batch()
    print(input.shape)
    print(input_len.shape)
    print(target.shape)
