# preprocess lines to data file
# data file stores a tuple (lines, seq_len, words, map)
# lines is a matrix of LINES x SEQLEN, zero padding, presenting words
# seq_len is a vector of length of each line
# words are list of words (character)
# map is a reverse map of words to index
# args: <txt> <output data file>

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

def data(path):
    with open(path, encoding="utf-8") as f:
        lines = [ l.strip() for l in f ]
    return preprocess(lines)

if __name__ == '__main__':
    out = data(sys.argv[1])
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(out, f)
