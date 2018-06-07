import pickle
import sys
import numpy as np
from util import tictoc

def check_equ(seq_a, seq_b):
    if len(seq_a) != len(seq_b):
        return False
    length = len(seq_a)
    for i in range(length):
        if seq_a[i] != seq_b[i]:
            return False
    return True

@tictoc('preprocess')
def preprocess3(src, des, k):
    with open(src, 'rb') as f:
        lines, lines_len, picks, maps = pickle.load(f, encoding='binary')
    lines_ = []
    lines_len_ = []
    for i in range(len(lines_len)):
        if i < k:
            continue
        temp = []
        for j in range(k+1):
            temp.append(lines[i-j])
        for j in range(1, k+1):
            if check_equ(temp[0], temp[j]) == True:
                break
            elif j == k:
                lines_.append(lines[i])
                lines_len_.append(lines_len[i])
    print(len(lines_))
    lines_ = np.array(lines_)
    lines_len_ = np.array(lines_len_, dtype=np.int32)
    with open(des, 'wb') as f:
        pickle.dump((lines_, lines_len_, picks, maps), f)

if __name__ == '__main__':
    preprocess3(sys.argv[1], sys.argv[2], int(sys.argv[3]))