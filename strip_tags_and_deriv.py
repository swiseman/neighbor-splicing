import sys

for line in sys.stdin:
    piece = line.strip().split('|||')[0]
    toks = piece.replace('</tgt>', ' </tgt>').replace('<tgt>', '<tgt> ').split()
    toks = toks[1:-1]
    if toks[0] == "<bos>":
        toks = toks[1:]
    if toks[-1] == "<eos>":
        toks = toks[:-1]
    print(" ".join(toks))
