import sys
import argparse
import math
import re
import os
from collections import Counter

import torch

from db2 import Dictionary
from utils import get_e2e_fields, get_wikibio_fields, dashrep

# from nltk...
stops = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
         "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
         "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
         "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
         "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
         "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
         "for", "with", "about", "against", "between", "into", "through", "during", "before",
         "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
         "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
         "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
         "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
         "will", "just", "don", "should", "now", ".", ",", "?", ";", ":", "-", "'", '"', "``",
         "''", "-lrb-", "-rrb-"}


def mask_tokes(tokes, fields, thestops):
    """
    masks subseq matches from the table; gonna be slow af
    """
    nutokes = tokes[:]
    for key, v in fields.items():
        vlow = [thing.lower() for thing in v]
        for k in range(len(vlow)):
            for l in range(len(vlow), k, -1):
                vlowkl = vlow[k:l]
                length = len(vlowkl)
                i = 0
                while i < len(tokes) - length + 1:
                    if ([toke.lower() for toke in tokes[i:i+length]] == vlowkl
                        and (len(vlowkl) > 1 or vlowkl[0] not in thestops)):
                        nutokes[i:i+length] = ["<mask>"]*length
                        i += length
                    else:
                        i += 1
    return nutokes

def mask_seq_tokes(tokes, srctokes, thestops):
    """
    assumes everything is already lower case
    masks subseq matches from the table
    """
    nutokes = tokes[:]
    for k in range(len(srctokes)):
        for l in range(len(srctokes), k, -1):
            vlow = srctokes[k:l]
            length = len(vlow)
            i = 0
            while i < len(tokes) - length + 1:
                if any(toke not in stops for toke in vlow) and tokes[i:i+length] == vlow:
                    nutokes[i:i+length] = ["<mask>"]*length
                    i += length
                else:
                    i += 1
    #return merge_mask_tokens(nutokes)
    return nutokes


parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, default="data/wb", help='datadir')
parser.add_argument('-vocsize', type=int, default=30000, help='')
parser.add_argument("-src_fi", default=None, type=str, help="")
parser.add_argument("-tgt_fi", default=None, type=str, help="")
parser.add_argument("-e2e", action='store_true', help="")
parser.add_argument('-min_count', type=int, default=1000, help='')
parser.add_argument("-split_dashes", action='store_true', help="")

if __name__ == "__main__":

    args = parser.parse_args()
    get_fields = get_e2e_fields if args.e2e else get_wikibio_fields

    thestops = stops

    #total_matches = 0
    with open(args.src_fi) as f1:
        with open(args.tgt_fi) as f2:
            for srcline in f1:
                tgtline = f2.readline()
                # if args.src_is_seq:
                #     srcs.append(line.strip().split())
                # else:
                fields = get_fields(srcline.strip().split()) # ordered key -> list dict
                if args.split_dashes:
                    fields = {k: re.sub(r'\w-\w', dashrep, " ".join(v)).split()
                              for k, v in fields.items()}
                    tgtline = re.sub(r'\w-\w', dashrep, tgtline)
                ttokes = tgtline.strip().split()

                nutokes = mask_tokes(ttokes, fields, thestops)
                print(" ".join(nutokes))
