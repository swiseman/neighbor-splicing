import os
import math
import re
import pickle

from collections import defaultdict, Counter

import numpy as np
import torch
#import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import tokenizers

from utils import get_wikibio_fields, get_e2e_fields, get_neidxs, dashrep
import cky

# TODO
# import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()},
#                                     language_level=3)
# import dbutils
import py_dbutils as dbutils

class Dictionary:
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.i2w = ["<pad>", unk_word, "<bos>", "<eos>", "<src>", "</src>", "<tgt>", "</tgt>"]
        self.w2i = {word: i for i, word in enumerate(self.i2w)}
        self.nspecial = len(self.i2w)

    def add_word(self, word, train=False):
        """
        returns idx of word
        """
        if train and word not in self.w2i:
            self.i2w.append(word)
            self.w2i[word] = len(self.i2w) - 1
        return self.w2i[word] if word in self.w2i else self.w2i[self.unk_word]

    def toks2idxs(self, toks):
        """
        does not require token to be in generatable vocab set
        """
        return [self.add_word(tok, train=False) for tok in toks]

    def bulk_add(self, words):
        """
        assumes train=True and no duplicates;
        """
        self.i2w.extend(words)
        #self.w2i = {word: i for i, word in enumerate(self.i2w)}
        self.w2i.update((wrd, len(self.w2i)) for wrd in words) # this seems to work

    def __len__(self):
        return len(self.i2w)


def relidx2abs(relidx, neidxs, missings, get_gen_idx):
    if relidx > len(neidxs): # a missing word
        absidx = -get_gen_idx(missings[relidx - len(neidxs) - 1]) # unk ok
    elif relidx == 0: # from src
        absidx = -2
    else: # a neighbor
        absidx = neidxs[relidx - 1]
    return absidx


def make_idxs_absolute(pptree, neidxs, missings, get_gen_idx):
    # replace relative idxs w/ absolute ones
    stack = [pptree]
    while stack:
        top = stack.pop()
        if top[0] != ('S',) and not isinstance(top[1], list): # a leaf
            absidx = relidx2abs(top[1], neidxs, missings, get_gen_idx)
            top[1] = absidx
        else:
            if top[0][0] == 'X': # has children
                absidx = relidx2abs(top[0][1], neidxs, missings, get_gen_idx)
                top[0][1] = absidx
            stack.extend(top[1])


class BaseDB(object):
    def __init__(self, args):
        if "wb" in args.data:
            self.min_field_count, self.max_voc_size = 100, 50000
            self.max_srclen, self.max_tgtlen = 130, 50
            self.dataset = "wb"
        elif "e2e" in args.data:
            self.min_field_count, self.max_voc_size = 1, 50000
            self.max_srclen, self.max_tgtlen = 100, 77 #70
            self.dataset = "e2e"
        else:
            assert False
        print("btw dataset:", self.dataset, "max_srclen", self.max_srclen, "max_tgtlen", self.max_tgtlen)
        self.enclose = args.enclose
        self.sel_firstlast_idxing = args.sel_firstlast_idxing

        missing_thresh, reg_thresh, max_gen_voc_size, max_voc_size = args.vocopts
        if missing_thresh is not None: # missing thresh
            assert reg_thresh is not None # reg thresh
            self.missing_thresh, self.reg_thresh = missing_thresh, reg_thresh
            self.max_gen_voc_size, self.max_voc_size = None, None
        elif reg_thresh is not None:
            self.missing_thresh, self.reg_thresh = None, reg_thresh
            self.max_gen_voc_size, self.max_voc_size = None, None
        else: # this option will break stuff
            if max_gen_voc_size is not None:
                self.max_gen_voc_size = max_gen_voc_size
            if max_voc_size is not None:
                self.max_voc_size = max_voc_size

        self.arbl = args.arbl # autoregressive baseline
        self.train_tgtfi = "masked-train-tgt.txt"
        #assert args.tokfi or self.dataset in ["e2e"]
        if args.tokfi:
            tokenizer = tokenizers.Tokenizer.from_file(args.tokfi)
            tokenizer.add_special_tokens(["<mask>"])
            self.tokenizer = tokenizer
            self.split_dashes = False
        else:
            self.tokenizer = None
            self.split_dashes = args.split_dashes
        self.get_vocab(self.train_tgtfi, args)

    def get_missing_counter(self, derivpath):
        if self.missing_thresh is not None:
            missingc = Counter()
            with open(derivpath, "rb") as f:
                while True:
                    try:
                        missings = pickle.load(f)[-1] # last element of tuple
                        missings = [thing[0] for thing in missings]
                        missingc.update(missings)
                    except EOFError:
                        break
        else:
            missingc = None
        return missingc

    def get_vocab(self, train_tgtfi, args):
        if os.path.exists(os.path.join(args.data, "dict.pt")):
            print("loading vocab...")
            self.d = torch.load(os.path.join(args.data, "dict.pt"))
        else:
            self.d = Dictionary()
            if self.dataset in ["wb", "e2e"]:
                self.make_vocab(os.path.join(args.data, "train-src.txt"),
                                os.path.join(args.data, train_tgtfi),
                                os.path.join(args.data, "train-encl-derivs.txt"))
            else:
                self.make_seq_vocab(os.path.join(args.data, "train-src.txt"),
                                    os.path.join(args.data, train_tgtfi),
                                    os.path.join(args.data, "train-encl-derivs.txt"))
            torch.save(self.d, os.path.join(args.data, "dict.pt"))

        self.get_gen_idx = lambda wrd: (self.d.w2i[wrd] if wrd in self.d.w2i
                                        and self.d.w2i[wrd] < self.d.gen_voc_size
                                        else self.d.w2i[self.d.unk_word])
        self.pad_idx = self.d.w2i["<pad>"]

    def make_seq_vocab(self, src_path, tgt_path, deriv_path):
        missingc = self.get_missing_counter(deriv_path)
        tgtc = Counter()
        with open(tgt_path) as f: # masked tgt, so already tokenized
            for line in f:
                tgtc.update(line.strip().split()[:self.max_tgtlen])
        if '' in tgtc:
            del tgtc['']

        if self.missing_thresh is not None: # then these are the whole genvoc
            self.d.bulk_add([wrd for wrd, count in missingc.items()
                             if wrd not in self.d.w2i and count >= self.missing_thresh])
            self.d.gen_voc_size = len(self.d)
            print("gen_voc_size/keepers", self.d.gen_voc_size)
            self.d.bulk_add([wrd for wrd, count in tgtc.items()
                             if wrd not in self.d.w2i and count >= self.reg_thresh])
            print("voc_size w/o src", len(self.d))
        elif self.reg_thresh is not None: # one threshold for everyone
            self.d.bulk_add([wrd for wrd, count in tgtc.items()
                             if wrd not in self.d.w2i and count >= self.reg_thresh])
            self.d.gen_voc_size = len(self.d)
            print("gen_voc_size/voc_size", len(self.d))
        else: # not based on frequency
            tgtvoc = tgtc.most_common(self.max_voc_size)
            if self.max_gen_voc_size is not None:
                self.d.bulk_add([wrd for wrd, _ in tgtvoc[:self.max_gen_voc_size]
                                 if wrd not in self.d.w2i])
                self.d.gen_voc_size = len(self.d)
                print("gen_voc_size", self.d.gen_voc_size)
                self.d.bulk_add([wrd for wrd, _ in tgtvoc[self.max_gen_voc_size:]
                                 if wrd not in self.d.w2i])
                print("voc_size", len(self.d))
            else:
                self.d.bulk_add([wrd for wrd, _ in tgtvoc if wrd not in self.d.w2i])
                self.d.gen_voc_size = len(self.d)
                print("gen_voc_size/voc_size", len(self.d))

        srcc = Counter()
        with open(src_path) as f:
            for line in f:
                if self.tokenizer:
                    srcc.update(self.tokenizer.encode(line.strip()).tokens[:self.max_srclen])
                else:
                    if self.split_dashes:
                        line = re.sub(r'\w-\w', dashrep, line.strip())
                    srcc.update(line.split()[:self.max_srclen])
        if '' in srcc:
            del srcc['']

        srcvoc = [key for key, count in srcc.items()
                  if key not in self.d.w2i and count >= self.min_src_count]
        self.d.bulk_add(srcvoc)
        print("full voc size", len(self.d))

    def make_vocab(self, src_path, tgt_path, deriv_path):
        missingc = self.get_missing_counter(deriv_path)
        fieldc, restc = Counter(), Counter()
        with open(src_path) as f:
            for line in f: # will just view as a sequence
                if self.dataset == "wb":
                    fields = get_wikibio_fields(line.strip().split()) # fieldname -> list of words
                else:
                    fields = get_e2e_fields(line.strip().split()) # fieldname -> list of words
                for key, wrdlist in fields.items():
                    fieldc[key] += 1
                    if self.tokenizer:
                        restc.update(self.tokenizer.encode(" ".join(wrdlist)).tokens)
                    else:
                        if self.split_dashes:
                            wrdlist = re.sub(r'\w-\w', dashrep, " ".join(wrdlist)).split()
                        restc.update(wrdlist)
        with open(tgt_path) as f: # already tokenized b/c masked
            for line in f:
                restc.update(line.strip().split()[:self.max_tgtlen])

        if '' in restc:
            del restc['']

        if self.missing_thresh is not None: # then these are the whole genvoc
            self.d.bulk_add([wrd for wrd, count in missingc.items()
                             if wrd not in self.d.w2i and count >= self.missing_thresh])
            self.d.gen_voc_size = len(self.d)
            print("gen_voc_size/keepers", self.d.gen_voc_size)
            self.d.bulk_add([wrd for wrd, count in restc.items()
                             if wrd not in self.d.w2i and count >= self.reg_thresh])
            print("voc_size w/o fields", len(self.d))
        elif self.reg_thresh is not None: # one threshold for everyone
            self.d.bulk_add([wrd for wrd, count in restc.items()
                             if wrd not in self.d.w2i and count >= self.reg_thresh])
            self.d.gen_voc_size = len(self.d)
            print("gen_voc_size/voc_size", len(self.d))
        else: # not based on frequency
            tgtvoc = restc.most_common(self.max_voc_size)
            if self.max_gen_voc_size is not None:
                self.d.bulk_add([wrd for wrd, _ in tgtvoc[:self.max_gen_voc_size]
                                 if wrd not in self.d.w2i])
                self.d.gen_voc_size = len(self.d)
                print("gen_voc_size", self.d.gen_voc_size)
                self.d.bulk_add([wrd for wrd, _ in tgtvoc[self.max_gen_voc_size:]
                                 if wrd not in self.d.w2i])
                print("voc_size", len(self.d))
            else:
                self.d.bulk_add([wrd for wrd, _ in tgtvoc if wrd not in self.d.w2i])
                self.d.gen_voc_size = len(self.d)
                print("gen_voc_size/voc_size", len(self.d))

        fieldvoc = [key for key, count in fieldc.items()
                    if key not in self.d.w2i and count >= self.min_field_count]
        self.d.bulk_add(fieldvoc)
        print("full voc size", len(self.d))

    def get_fullderiv(self, idx, val=False):
        """
        this is just to look at stuff
        """
        if val:
            moves_list, srcs_list = self.val_moves, self.val_srcs
        else:
            moves_list, srcs_list = self.train_moves, self.train_srcs

        canvases = [["<tgt>", "</tgt>"]]
        lastcanv = canvases[0]
        for m in range(len(moves_list[idx])):
            move = moves_list[idx][m]
            neidx, l, r, jj, skip = move # N.B. neidx is now absolute
            if neidx >= 0:
                span = self.train_tgts[neidx][l:r]
            elif neidx == -2:
                span = srcs_list[idx][l:r]
            else:
                span = [-neidx]
            assert len(span) == r-l
            # increment jjs b/c of <tgt>
            canvas = lastcanv[:jj+1] + [self.d.i2w[wrd] for wrd in span] + lastcanv[jj+skip+1:]
            canvases.append(canvas)
            lastcanv = canvas

        return canvases

    def tokenize_things(self, datadir, srcfi, tgtfi, derivfi, keepwrds=False):
        srcs = []
        if srcfi is not None:
            with open(os.path.join(datadir, srcfi)) as f:
                for line in f:
                    if self.dataset in ["wb", "e2e"]:
                        # will just view as a sequence
                        if self.dataset == "wb":
                            fields = get_wikibio_fields(line.strip().split())
                        else:
                            fields = get_e2e_fields(line.strip().split())
                        src = []
                        for k in fields.keys():
                            v = fields[k]
                            src.append(k)
                            if self.tokenizer:
                                src.extend(self.tokenizer.encode(" ".join(v)).tokens)
                            else:
                                if self.split_dashes:
                                    v = re.sub(r'\w-\w', dashrep, " ".join(v)).split()
                                src.extend(v)
                    else:
                        if self.tokenizer:
                            src = self.tokenizer.encode(line.strip()).tokens
                        else:
                            if self.split_dashes:
                                line = re.sub(r'\w-\w', dashrep, line)
                            src = line.strip().split()

                    src = src[:self.max_srclen]
                    if not keepwrds:
                        #src = torch.IntTensor(self.d.toks2idxs(src))
                        src = self.d.toks2idxs(src)
                    srcs.append(src)

        tgts = []
        if tgtfi is not None:
            with open(os.path.join(datadir, tgtfi)) as f:
                for line in f:
                    tgt = line.strip().split()[:self.max_tgtlen] # already tokenized b/c masked
                    if not keepwrds:
                        #tgt = torch.IntTensor(self.d.toks2idxs(tgt))
                        tgt = self.d.toks2idxs(tgt)
                    if self.enclose:
                        bostok = "<bos>" if keepwrds else self.d.w2i["<bos>"]
                        eostok = "<eos>" if keepwrds else self.d.w2i["<eos>"]
                        tgts.append([bostok] + tgt + [eostok])
                    else:
                        tgts.append(tgt)

        moveseqs = []
        if derivfi is not None:
            derividx = 0
            isval = tgtfi is None
            all_neidxs = self.train_neidxs if not isval else self.val_neidxs
            with open(os.path.join(datadir, derivfi), 'rb') as f:
                while True:
                    try:
                        tup = pickle.load(f)
                        if len(tup) == 3:
                            pptree, neidxs, missings = tup
                            if pptree[0][0] == 'insert': # a greedy thing...
                                pptree = ["S", [list(thing) + [0] for thing in pptree]]
                        elif len(tup) == 2:
                            pptree, missings = tup
                            if pptree[0][0] == 'insert': # a greedy thing...
                                pptree = ["S", [list(thing) + [0] for thing in pptree]]
                            neidxs = all_neidxs[derividx].tolist()
                            neset_i = set(neidxs)
                            if not isval:
                                neset_i.add(derividx)
                            neidxs.extend([pidx for pidx in self.protes if pidx not in neset_i])
                        else:
                            assert False
                        # missings is a list of 1-lists for some stupid reason
                        missings = [thing[0] for thing in missings]
                        make_idxs_absolute(pptree, neidxs, missings, self.get_gen_idx)
                        moveseqs.append(pptree)
                        derividx += 1
                    except EOFError:
                        break
                assert derividx == len(srcs)

        return srcs, tgts, moveseqs

    def get_data(self, train_tgtfi, train_derivfi, val_srcfi, val_derivfi, val_nefi,
                 tokpath, args, keepwrds=False, val=False):
        # if not val:
        #     self.train_neidxs = get_neidxs(args.data, "train-nes.txt", args.nne)
        # self.val_neidxs = get_neidxs(args.data, val_nefi, args.nne, val=True)
        if os.path.exists(os.path.join(args.data, tokpath)):
            print("loading tokenized stuff...")
            if self.arbl:
                (self.train_srcs, self.train_tgts, self.train_rultgts, self.val_srcs,
                 self.val_tgts, self.val_rultgts) = torch.load(os.path.join(args.data, tokpath))
            else:
                (self.train_srcs, self.train_tgts, self.train_moves,
                 self.val_srcs, self.val_moves) = torch.load(os.path.join(args.data, tokpath))
        else:
            if self.arbl:
                self.train_srcs, self.train_tgts, _ = self.tokenize_things(
                    args.data, "train-src.txt", self.train_tgtfi, None, keepwrds=keepwrds)
                valtgtfi = None if val else "val-tgt.txt"
                self.val_srcs, self.val_tgts, _ = self.tokenize_things(
                    args.data, val_srcfi, valtgtfi, None, keepwrds=keepwrds)
                # need to also get nonmasked tgts as input
                self.train_rultgts, self.val_rultgts = [], []
                if not val:
                    rultgtses = [self.train_rultgts, self.val_rultgts]
                    rulfis = ["train-tgt.txt", "val-tgt.txt"]
                    for r, rulfi in enumerate(rulfis):
                        with open(os.path.join(args.data, rulfi)) as f:
                            for line in f:
                                if self.tokenizer:
                                    rultoks = self.tokenizer(line.strip()).tokens[:self.max_tgtlen]
                                else:
                                    if self.split_dashes:
                                        line = re.sub(r'\w-\w', dashrep, line)
                                    rultoks = line.strip().split()[:self.max_tgtlen]
                                rultgtses[r].append(rultoks)

                torch.save((self.train_srcs, self.train_tgts, self.train_rultgts, self.val_srcs,
                            self.val_tgts, self.val_rultgts), os.path.join(args.data, tokpath))
            else:
                train_srcfi = None if val else "train-src.txt"
                self.train_srcs, self.train_tgts, self.train_moves = self.tokenize_things(
                    args.data, train_srcfi, self.train_tgtfi, train_derivfi, keepwrds=keepwrds)
                self.val_srcs, self.val_tgts, self.val_moves = self.tokenize_things(
                    args.data, val_srcfi, None, val_derivfi, keepwrds=keepwrds)
                torch.save((self.train_srcs, self.train_tgts, self.train_moves,
                            self.val_srcs, self.val_moves), os.path.join(args.data, tokpath))

    def get_padded_srcs(self, batch, val=False):
        srcs_list = self.val_srcs if val else self.train_srcs
        if self.sel_firstlast_idxing:
            #srcs = pad_sequence([srcs_list[idx].long() for idx in batch],
            srcs = pad_sequence([torch.LongTensor(srcs_list[idx]) for idx in batch],
                                padding_value=self.pad_idx)
        else:
            assert False
            srcs = pad_sequence([torch.cat([torch.LongTensor([self.d.w2i["<src>"]]),
                                            srcs_list[idx].long(),
                                            torch.LongTensor([self.d.w2i["</src>"]])], 0)
                                 for idx in batch], padding_value=self.pad_idx)
        return srcs

    def get_padded_nes(self, neidxs):
        if self.sel_firstlast_idxing:
            #nes = pad_sequence([self.train_tgts[idx].long() for idx in neidxs],
            nes = pad_sequence([torch.LongTensor(self.train_tgts[idx]) for idx in neidxs],
                               padding_value=self.pad_idx)
        else:
            nes = pad_sequence([torch.cat([torch.LongTensor([self.d.w2i["<tgt>"]]),
                                           self.train_tgts[idx].long(),
                                           torch.LongTensor([self.d.w2i["</tgt>"]])], 0)
                                for idx in neidxs], padding_value=self.pad_idx)
        return nes


class TrainDB(BaseDB):
    def __init__(self, args):
        super().__init__(args)
        tokpath = "encl_tok_data.pt" if self.enclose else "tok_data.pt"
        if args.leftright:
            tokpath = "gr_" + tokpath
        if args.arbl:
            tokpath = "ar_" + tokpath

        train_derivfi, val_derivfi = "derivs.txt", "derivs.txt"
        if args.leftright:
            train_derivfi, val_derivfi = "gr-" + train_derivfi, "gr-" + val_derivfi
        if args.enclose:
            train_derivfi, val_derivfi = "encl-" + train_derivfi, "encl-" + val_derivfi
        train_derivfi, val_derivfi = "train-" + train_derivfi, "val-" + val_derivfi

        val_srcfi, val_nefi = "val-src.txt", "val-nes.txt"

        if args.prote_fi:
            try:
                self.protes = torch.load(args.prote_fi)
            except pickle.UnpicklingError:
                self.protes = np.load(args.prote_fi)
        else:
            self.protes = set([])

        self.train_neidxs = get_neidxs(args.data, "train-nes.txt", args.nne)
        self.val_neidxs = get_neidxs(args.data, val_nefi, args.nne, val=True)
        self.get_data(self.train_tgtfi, train_derivfi, val_srcfi, val_derivfi,
                      val_nefi, tokpath, args, keepwrds=args.arbl)

        if not isinstance(self.protes, set):
            self.protes = set(self.protes)

        if not args.arbl:
            self.flat_moves = args.flat_moves
            if self.flat_moves: # flatten move trees into move lists
                for moveslist in [self.train_moves, self.val_moves]:
                    for i in range(len(moveslist)):
                        temp = []
                        cky.movesfromtree(moveslist[i], temp)
                        moveslist[i] = [move[1:] for move in temp] # remove nonterm

            # we only need to keep around idxs that aren't in the moves
            print("calcing remaining...")
            self.train_remaining, self.val_remaining = [], []
            for i in range(len(self.train_neidxs)):
                #self.train_neidxs[i] = set(self.train_neidxs[i]) | self.protes
                _, netgts, _, _ = self.get_canvases(i, 2, max_canvlen=args.max_canvlen, val=False)
                nes_i = {move[0] for netgtsi in netgts for move in netgtsi
                         if move[0] is not None and move[0] >= 0}
                nes_i.add(i) # don't want to use same example if it's a prote
                remaining = (set(self.train_neidxs[i].numpy()) | self.protes) - nes_i
                self.train_remaining.append(torch.IntTensor(list(remaining)))
            del self.train_neidxs
            for i in range(len(self.val_neidxs)):
                #self.val_neidxs[i] = set(self.val_neidxs[i]) | self.protes
                _, netgts, _, _ = self.get_canvases(i, 2, max_canvlen=args.max_canvlen, val=True)
                nes_i = {move[0] for netgtsi in netgts for move in netgtsi
                         if move[0] is not None and move[0] >= 0}
                remaining = (set(self.val_neidxs[i].numpy()) | self.protes) - nes_i
                self.val_remaining.append(torch.IntTensor(list(remaining)))
            del self.val_neidxs
            print("done with remaining...")
        else:
            for i in range(len(self.train_rultgts)):
                self.train_rultgts[i] = ["<bos>"] + self.train_rultgts[i] + ["<eos>"]
            for i in range(len(self.val_rultgts)):
                self.val_rultgts[i] = ["<bos>"] + self.val_rultgts[i] + ["<eos>"]

        # this is gross, but whatever
        self.bsz, self.val_bsz = args.bsz, args.val_bsz
        self.nbatches = int(math.ceil(len(self.train_srcs)/self.bsz))
        self.bidx = self.nbatches # will get reset
        self.nval_batches = int(math.ceil(len(self.val_srcs)/self.val_bsz))
        self.val_bidx = self.nval_batches
        self.val_perm = torch.arange(len(self.val_srcs))

    def next_batch(self):
        if self.bidx >= self.nbatches:
            self.perm = torch.randperm(len(self.train_srcs))
            self.bidx = 0
        idxs = self.perm[self.bidx*self.bsz:(self.bidx+1)*self.bsz]
        self.curr_batch = idxs
        self.bidx += 1
        return idxs

    def next_val_batch(self):
        if self.val_bidx >= self.nval_batches:
            self.val_bidx = 0
        idxs = self.val_perm[self.val_bidx*self.val_bsz:(self.val_bidx+1)*self.val_bsz]
        self.curr_val_batch = idxs
        self.val_bidx += 1
        return idxs

    def next_state(self, lastcanv, lastrelidx, lastfeats, move, mvidx, src):
        inc = int(not self.sel_firstlast_idxing)
        nufeats = lastfeats.clone()
        neidx, l, r, jj, skip = move # N.B. neidx is now absolute
        if neidx >= 0:
            span = self.train_tgts[neidx][l:r]
        elif neidx == -2:
            span = src[l:r]
            nufeats[l+inc:r+inc].fill_(1) # inc b/c of <src>
        else:
            span = [-neidx]
        if len(span) != r-l:
            print(neidx, l, r, len(span))
        assert len(span) == r-l
        # increment jjs b/c of <tgt>
        canvas = lastcanv[:jj+1] + span + lastcanv[jj+skip+1:]
        #1-based mv idxing
        relidx = lastrelidx[:jj+1] + [mvidx+1]*(r-l) + lastrelidx[jj+skip+1:]
        return canvas, relidx, nufeats

    def get_canvases(self, idx, srclen, max_canvlen=10000, val=False):
        if val:
            moves_list, srcs_list = self.val_moves, self.val_srcs
        else:
            moves_list, srcs_list = self.train_moves, self.train_srcs
        boc, eoc = self.d.w2i["<tgt>"], self.d.w2i["</tgt>"]
        if self.flat_moves:
            return dbutils.get_move_canvases(
                moves_list, srcs_list, idx, srclen, self.next_state, boc, eoc,
                max_canvlen=max_canvlen, ignore_unk=(self.d.gen_voc_size == self.d.nspecial))
        return dbutils.get_tree_canvases(
            moves_list, srcs_list, idx, srclen, self.next_state, boc, eoc,
            max_canvlen=max_canvlen)

    def do_roll_in(self, min_nes, max_canvlen=10000, val=False, leftright=False):
        """
        returns:
            (max_canv_len x bsz, bsz, max_sel_tgt_len x bsz) tensors
        """
        if val:
            batch, srcs_list = self.next_val_batch(), self.val_srcs
            remaining_idxs = self.val_remaining
        else:
            batch, srcs_list = self.next_batch(), self.train_srcs
            remaining_idxs = self.train_remaining

        srcs = self.get_padded_srcs(batch, val=val)
        alcanvases, alnetgts, alrelidxs, alsrcfeats, neidxs = [], [], [], [], []

        canv2ii = []
        for ii, bidx in enumerate(batch):
            canvases, netgts, mvidxs, srcfeats = self.get_canvases(
                bidx, srcs.size(0), max_canvlen=max_canvlen, val=val)
            alcanvases.extend([torch.LongTensor(canvas) for canvas in canvases])
            alnetgts.extend(netgts)
            canv2ii.extend([ii]*len(netgts))
            alrelidxs.extend([torch.LongTensor(mvidxsj) for mvidxsj in mvidxs])
            alsrcfeats.extend(srcfeats)
            nes_ii = {move[0] for netgtsii in netgts for move in netgtsii
                      if move[0] is not None and move[0] >= 0}
            # remaining = list(nes_sets[bidx] - nes_ii)
            remaining = remaining_idxs[bidx]
            neidxs.extend(nes_ii)
            neperm = torch.randperm(len(remaining))
            neidxs.extend([remaining[neperm[j]].item() for j in range(min_nes - len(nes_ii))])

        alcanvases = pad_sequence(alcanvases, padding_value=self.pad_idx)
        alrelidxs = pad_sequence(alrelidxs, padding_value=-1)
        alsrcfeats = pad_sequence(alsrcfeats, padding_value=3)
        assert alcanvases.size(1) == alrelidxs.size(1)
        assert alcanvases.size(1) == alsrcfeats.size(1)
        assert alcanvases.size(1) == len(alnetgts)
        # repeat srcs as necessary
        srcs = srcs.index_select(1, torch.LongTensor(canv2ii))
        assert alcanvases.size(1) == srcs.size(1)

        neidxs = list(set(neidxs)) # uniquify in case multiple examples w/ same neighbs
        self.curr_neidxs = neidxs # h4ck
        nes = self.get_padded_nes(neidxs)

        uni_locs = defaultdict(list) # wrd -> list of (neidx, start)
        [uni_locs[wrd].append((kk, start)) for kk, neidx in enumerate(neidxs)
         for start, wrd in enumerate(self.train_tgts[neidx])]
        ne2neidx = {neidx: kk for kk, neidx in enumerate(neidxs)}

        nne, nelen, srclen = len(neidxs), nes.size(0), srcs.size(0)
        fin_idx = nelen*nne+self.d.w2i["<src>"] # done idx matches canv's <tgt> w/ unused <src>
        starttgts, ralnetgts = dbutils.get_startend_tgts(
            alnetgts, nelen, nne, fin_idx, self.d.gen_voc_size, srclen, batch, canv2ii,
            self.train_tgts, ne2neidx, uni_locs, neidxs, srcs_list, self.d.w2i["<unk>"],
            val, leftright=leftright)
        starttgts = pad_sequence(starttgts, padding_value=-1) # 0'th column is dummy
        return srcs, alsrcfeats, nes, (alcanvases, alrelidxs), starttgts, ralnetgts, fin_idx

    def do_bl_batch(self, min_nes, max_canvlen=10000, val=False):
        """
        returns:
            (max_canv_len x bsz, bsz, max_sel_tgt_len x bsz) tensors
        """
        if val:
            batch, srcs_list = self.next_val_batch(), self.val_srcs
            neidx_list, tgts_list = self.val_neidxs, self.val_rultgts
        else:
            batch, srcs_list = self.next_batch(), self.train_srcs
            neidx_list, tgts_list = self.train_neidxs, self.train_rultgts

        srcs = pad_sequence( # srclen x bsz
            [torch.LongTensor(self.d.toks2idxs(srcs_list[bidx]))
             for bidx in batch], padding_value=self.pad_idx)
        tgtinps = pad_sequence( # tgtlen x bsz
            [torch.LongTensor(self.d.toks2idxs(tgts_list[bidx][:-1]))
             for bidx in batch], padding_value=self.pad_idx)

        if min_nes > 0:
            neidxs = set()
            for bidx in batch:
                if min_nes < len(neidx_list[bidx]):
                    neperm = torch.randperm(len(neidx_list[bidx]))[:min_nes]
                    neidxs.update([neidx_list[bidx][pidx].item() for pidx in neperm])
                else:
                    neidxs.update(neidx_list[bidx].numpy())
            neidxs = list(neidxs)
            nes = pad_sequence( # nelen x nne
                [torch.LongTensor(self.d.toks2idxs(self.train_tgts[neidx]))
                 for neidx in neidxs], padding_value=self.pad_idx)
            nelen, nne = nes.size()

            # get locations among masked neighbors
            type2loc = defaultdict(list) # word type -> index among neighbors
            [type2loc[wrd].append(t*nne+n)
             for n, neidx in enumerate(neidxs) for t, wrd in enumerate(self.train_tgts[neidx])]
            assert "<eos>" not in type2loc
        else:
            type2loc, nes, nelen, nne = {}, torch.LongTensor(), 0, 0

        offset = nelen*nne
        srctype2locs = []
        for bidx in batch:
            srctype2loc_b = defaultdict(list)
            [srctype2loc_b[wrd].append(offset + self.d.gen_voc_size + s)
             for s, wrd in enumerate(srcs_list[bidx])]
            srctype2locs.append(srctype2loc_b)

        tgtidxs = [] # like in segment case, we only generate if can't copy
        for t in range(1, tgtinps.size(0)+1): # ignore <bos> token
            for b, bidx in enumerate(batch):
                if t < len(tgts_list[bidx]):
                    wrd = tgts_list[bidx][t]
                    tbidxs = []
                    if wrd in srctype2locs[b]:
                        tbidxs.extend(srctype2locs[b][wrd])
                    if wrd in type2loc:
                        tbidxs.extend(type2loc[wrd])
                    if not tbidxs: # generate; should include <eos>
                        tbidxs = [offset + self.get_gen_idx(wrd)]
                    tgtidxs.append(torch.LongTensor(tbidxs))
                else:
                    tgtidxs.append(torch.LongTensor([-1]))

        tgtidxs = pad_sequence(tgtidxs, batch_first=True, padding_value=-1) # T*bsz x maxtgts
        return srcs, nes, tgtinps, tgtidxs


# used at test time; stores real words.
class ValDB(BaseDB):
    def __init__(self, args):
        assert os.path.exists(os.path.join(args.data, "dict.pt"))
        super().__init__(args)
        val_srcfi = "val-src.txt" if args.val_src_fi is None else args.val_src_fi
        print("using val src", val_srcfi)
        print("getting val neighbors from", args.val_nefi)

        tokpath = "encl_kw_data.pt" if self.enclose else "kw_data.pt"
        tokpath = val_srcfi.split('.')[0].split('-')[0] + "_" + tokpath

        if args.arbl:
            tokpath = "ar_" + tokpath

        #neidxs, nedists = get_neidxs(args.data, args.val_nefi, args.nne, val=True, dist=True)
        neidxs = get_neidxs(args.data, args.val_nefi, args.nne, val=True, dist=False)
        neidxs = neidxs.tolist()
        nedists = [[None] for _ in range(len(neidxs))]

        self.get_data(self.train_tgtfi, None, val_srcfi, None, args.val_nefi,
                      tokpath, args, keepwrds=True, val=True)
        if args.prote_fi:
            try:
                protes = torch.load(args.prote_fi)
            except pickle.UnpicklingError:
                protes = np.load(args.prote_fi)
            if not isinstance(protes, set):
                protes = set(protes)
        else:
            print("not using protes...")
            protes = set([])

        # easier if neighbors are a list
        self.val_neidxs, self.val_ne_dists = [], []
        for i in range(len(neidxs)):

            self.val_neidxs.append(list(protes))
            self.val_neidxs[-1].extend([neidx for neidx in neidxs[i] if neidx not in protes])
            self.val_ne_dists.append(nedists[i][0]) # just min dist

        assert len(self.val_neidxs) == len(neidxs)
        assert len(self.val_ne_dists) == len(neidxs)
        self.protes = protes
