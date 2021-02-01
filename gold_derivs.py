"""
different than mgd2 in that we're predicting <bos> tgt <eos> and also wrapping neighbors
in bos and eoses, to encourage copying from a neighbor first..
"""
import os
import argparse
import pickle
import re
import numpy as np
import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()},
                                    language_level=3)

from collections import OrderedDict

from utils import get_wikibio_fields, get_e2e_fields, dashrep
import torch
import tokenizers
#from nudb import Pants, mock_reconstruct


def fixup_for_table(moves, fieldtups, finalnes, istree=False):
    """
    collapse table entries into a single src seq and update everything accordingly
    fieldtups - list of [key, vallist] from src
    """
    nsrcs = len(fieldtups)
    csrc = [] # collapsed src
    src2offset = {} # maps j'th src entry to idx of the first word in j's vallist within csrc
    for j, (key, vallist) in enumerate(fieldtups):
        csrc.append(key)
        src2offset[j] = len(csrc)
        csrc.extend(vallist)

    if istree: # should modify inplace
        numoves = moves
        stack = [child for child in numoves[1]] # root's children
        while stack:
            top = stack.pop()
            if isinstance(top[1], list): # has children
                cnode = top[0]
            else:
                cnode = top
            neidx, l, r = cnode[1:4]
            if neidx >= nsrcs:
                cnode[1] = neidx-nsrcs+1
            else:
                offset = src2offset[neidx]
                cnode[1] = 0
                cnode[2] = offset + l
                cnode[3] = offset + r
            if isinstance(top[1], list):
                stack.extend(top[1])
    else:
        numoves = []
        for move in moves:
            thing, neidx, l, r, ii, skip = move
            if neidx >= nsrcs: # not from src; just subtract
                numoves.append((thing, neidx-nsrcs+1, l, r, ii, skip))
            else: # from src; need to figure out actual location
                offset = src2offset[neidx]
                numoves.append((thing, 0, offset+l, offset+r, ii, skip))

    nufinalnes = [csrc]
    nufinalnes.extend(finalnes[nsrcs:])
    return numoves, nufinalnes


def get_src(line, tokenizer, get_fields, src_is_seq=False, split_dashes=False):
    if src_is_seq:
        if tokenizer:
            stokes = tokenizer.encode(line.strip()).tokens
        else:
            stokes = line.strip().split()
        return stokes[:args.max_srclen]
    fields = get_fields(line.strip().split()) # ordered key -> list dict
    # truncate so num_keys + num_vals doesn't exceed max_srclen
    truncfields, len_so_far = [], 0
    for k, v in fields.items():
        vallist = []
        truncfields.append((k, vallist))
        len_so_far += 1 # for k; assuming always one token!
        if len_so_far >= args.max_srclen:
            break
        if split_dashes:
            v = re.sub(r'\w-\w', dashrep, " ".join(v)).split()
        elif tokenizer:
            v = tokenizer.encode(" ".join(v)).tokens
        vallist.extend(v[:args.max_srclen-len_so_far])
        len_so_far += len(vallist)
        if len_so_far >= args.max_srclen:
            break
    return truncfields

parser = argparse.ArgumentParser()
parser.add_argument("-val_src_fi", default=None, type=str, help="")
parser.add_argument("-val_tgt_fi", default=None, type=str, help="")
parser.add_argument("-val_ne_fi", default=None, type=str, help="")
parser.add_argument("-ne_tgt_fi", default=None, type=str, help="tgt neighbors (typically from train)")
parser.add_argument("-out_fi", default=None, type=str, help="")
parser.add_argument("-tokfi", default=None, type=str, help="")
parser.add_argument('-bsz', type=int, default=200, help='')
parser.add_argument('-nne', type=int, default=100, help='')
parser.add_argument("-wrkr", default="1,1", type=str, help="")
parser.add_argument("-slow", action='store_true', help="")
parser.add_argument("-val", action='store_true', help="don't worry about self copying..")
parser.add_argument("-just_cat", action='store_true', help="")
parser.add_argument("-enclose", action='store_true', help="enclose tgts w/ bos, eos")
parser.add_argument("-src_is_seq", action='store_true', help="")
parser.add_argument("-e2e", action='store_true', help="")
parser.add_argument("-prote_fi", default="", type=str, help="")
parser.add_argument('-max_srclen', type=int, default=120, help='')
parser.add_argument('-max_tgtlen', type=int, default=50, help='')
parser.add_argument('-skip', type=int, default=0, help='')
parser.add_argument("-split_dashes", action='store_true', help="")

if __name__ == "__main__":

    args = parser.parse_args()

    print(args)

    if args.slow:# or args.just_cat:
        import cky
    else:
        import nucky as cky

    wid, nwrkrs = args.wrkr.split(',')
    wid, nwrkrs = int(wid), int(nwrkrs)

    assert not os.path.exists(args.out_fi + "-" + str(wid))

    get_fields = get_e2e_fields if args.e2e else get_wikibio_fields

    if args.tokfi:
        tokenizer = tokenizers.Tokenizer.from_file(args.tokfi)
        tokenizer.add_special_tokens(["<mask>"])
        print("using tokenizer from", args.tokfi)
    else:
        tokenizer = None

    netgts = []
    with open(args.ne_tgt_fi) as f:
        for line in f: # should already be tokenized
            ttokes = line.strip().split()
            if args.enclose:
                netgts.append(["<bos>"] + ttokes[:args.max_tgtlen] + ["<eos>"])
            else:
                netgts.append(ttokes[:args.max_tgtlen])

    # get neidxs for each tgt sentence
    neidxs = []
    with open(args.val_ne_fi) as f:
        for l, line in enumerate(f):
            if args.val: # using training neighbors for validation
                netups = line.strip().split()[:args.nne]
                idxs = [int(thing.split(',')[0]) for thing in netups]
            else:
                netups = line.strip().split()[:args.nne+1] # not sure if self is inside
                idxs = [int(thing.split(',')[0]) for thing in netups]
                idxs = [thing for thing in idxs if thing != l][:args.nne]
            neidxs.append(idxs)

    # if len(neidxs) != len(tgts):
    #     print("umm just doing the ones we have neighbors for, bro")
    #assert len(neidxs) == len(tgts)

    if args.prote_fi:
        protes = torch.load(args.prote_fi)

    # just get chunk for this worker
    if nwrkrs > 1:
        chunksz = int(len(neidxs)/nwrkrs)
        start = (wid-1)*chunksz
        end = wid*chunksz if wid < nwrkrs else len(neidxs)
        #start += 119285
    else:
        start, end = 0, len(neidxs)

    start += args.skip

    STARTNT = ('S',) if args.slow else (-3,) # for parse
    #STARTNT = ('S0',) if args.slow else (-5,) # for parse2

    print("use_protes", args.prote_fi)

    # get actions somehow
    with open(args.out_fi + "-" + str(wid), "ab") as f:
        with open(args.val_src_fi) as fsrc:
            with open(args.val_tgt_fi) as ftgt:
                #for i in range(start, end):
                for i, srcline in enumerate(fsrc):
                    tgtline = ftgt.readline()
                    if i >= start and i < end:
                        print("doing", i)
                        src = get_src(srcline, tokenizer, get_fields,
                                      split_dashes=args.split_dashes, src_is_seq=args.src_is_seq)
                        if tokenizer:
                            tgt = tokenizer.encode(tgtline.strip()).tokens[:args.max_tgtlen]
                        elif args.split_dashes:
                            tgt = re.sub(
                                r'\w-\w', dashrep, tgtline.strip()).split()[:args.max_tgtlen]
                        else:
                            tgt = tgtline.strip().split()[:args.max_tgtlen]
                        #import ipdb; ipdb.set_trace()
                        if args.enclose:
                            tgt = ["<bos>"] + tgt + ["<eos>"]

                        if not args.src_is_seq:
                            srctups = src # list of (key, val-list)
                            if not srctups[-1][1]: # can only have an empty list at the end
                                fieldtups = srctups[:-1]
                            else:
                                fieldtups = srctups
                            # only last vallist can be empty
                            nes_i = [vallist for key, vallist in fieldtups]
                        else:
                            nes_i = [src] # start w/ src too
                        if args.prote_fi:
                            neset_i = set(neidxs[i])
                            if not args.val:
                                neset_i.add(i) # so that we don't add ourselves
                            neidxs[i].extend([neidx for neidx in protes if neidx not in neset_i])
                        nes_i.extend([netgts[neidx] for neidx in neidxs[i]])
                        # figure out what unigrams we're missing
                        neunis = set()
                        [neunis.update(ne) for ne in nes_i]
                        missing_unis = set(tgt) - neunis
                        # add pseudo-sentences for each missing unigram
                        missingnes = [[thing] for thing in sorted(missing_unis)]
                        finalnes = nes_i + missingnes
                        if args.just_cat:
                            moves = cky.greedy_tag(tgt, finalnes)
                            if not args.src_is_seq:
                                # not using last empty field if there since it won't matter
                                moves, finalnes = fixup_for_table(moves, fieldtups, finalnes)
                            rec = cky.reconstruct(moves, finalnes)
                            if rec != tgt:
                                print("wtf", rec, tgt, moves)
                            assert rec == tgt
                            pptree = moves # so it gets saved
                        else:
                            # run cky
                            try:
                                table, bps, _ = cky.parse(tgt, finalnes, 9500)
                            except IndexError:
                                try:
                                    table, bps, _ = cky.parse(tgt, finalnes, 10500)
                                except IndexError:
                                    print("skipping", i)
                                    assert False
                                    bps = None
                            if bps is not None:
                                # back out the derivation
                                tree = cky.backtrack(STARTNT, finalnes, bps, 0, len(tgt))
                                # read off the tree leaves and get the moves
                                #moves = cky.get_moves(tree)
                                pptree = cky.get_movetree(tree)
                                if not args.src_is_seq:
                                    # not using last empty field if there since it won't matter
                                    #moves, finalnes1 = fixup_for_table
                                    #    (moves, fieldtups, finalnes, istree=False)
                                    pptree, finalnes1 = fixup_for_table(
                                        pptree, fieldtups, finalnes, istree=True)
                                moves2 = []
                                cky.movesfromtree(pptree, moves2)
                                rec = cky.reconstruct(moves2, finalnes1)
                                if rec != tgt:
                                    print("wtf", rec, tgt, moves)
                                assert rec == tgt

                            else:
                                moves = ["SKIP"]
                        pickle.dump((pptree, neidxs[i], missingnes), f, pickle.DEFAULT_PROTOCOL)
