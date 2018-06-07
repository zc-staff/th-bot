from config import *
from data1 import Batches as Batches1

class Batches(Batches1):
    def __init__(self, lines, seq_len):
        super().__init__(lines, seq_len)
        self.size -= 1
        self.perm = np.random.permutation(self.size)

    def next_batch(self):
        incides = self.next_incides()
        encoder_input = self.lines[incides, 0:SEQLEN]
        encoder_len = self.seq_len[incides]
        decoder_input = self.lines[1 + incides, 0:SEQLEN]
        decoder_len = self.seq_len[1 + incides]
        target = self.lines[1 + incides, 1:SEQLEN + 1]
        return encoder_input, encoder_len, decoder_input, decoder_len, target
