# preprocess raw qq txt to lines
# filter special messages
# args: <raw txt> <output file> 

import sys
import re

pat1 = '[0-9]+-[0-9]+-[0-9]+ [0-9]+:[0-9]+:[0-9]+ '
pat2 = r'\[图片\]|@|\[表情\]|\[发起投票\]'
pat3 = r'https?://.*$'
pat4 = r'^.+撤回了一条消息$'
pat4 = r'^.+礼物\].*$'

with open(sys.argv[1]) as f:
    lines = [ l.strip() for l in f ]

lines = filter(lambda x : len(x) > 0 and re.search(pat1, x) is None, lines)
lines = map(lambda x : re.sub(pat2, '', x), lines)
lines = map(lambda x : re.sub(pat3, '', x), lines)
lines = map(lambda x : re.sub(pat4, '', x), lines)
lines = filter(lambda x : len(x) > 0, lines)
lines = map(lambda x : x + '\n', lines)

with open(sys.argv[2], 'w') as f:
    f.writelines(lines)
