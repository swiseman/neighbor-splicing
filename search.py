import os
import argparse
from collections import Counter, OrderedDict

import torch

import db2 as data
import model_utils as mutils
import se_search
from cky import movesfromtree

def get_thresh_stuff(mindist, threshes):
    for d, vals in threshes.items():
        if mindist <= d:
            return vals
    assert False

def batched_pred(db, model, device, args, restrict_fn=None):
    if restrict_fn is not None:
        for ii in range(len(db.val_neidxs)):
            restricted_nes = restrict_fn(db.val_neidxs[ii], db, ii)
            db.val_neidxs[ii] = restricted_nes[:args.restrict_nes]
            if len(db.val_neidxs[ii]) < args.restrict_nes:
                print("not enough restricted neighbors for", ii, len(db.val_neidxs[ii]))

    nelists, nedists, nprotes = db.val_neidxs, db.val_ne_dists, len(db.protes)
    model.eval()
    num_preds = len(db.val_srcs)
    print("num_preds", num_preds, len(nelists))
    print("nprotes", nprotes)
    #assert len(nelists) == num_preds

    if args.debug >= 0:
        start, end = args.debug, args.debug+1
    elif args.startend:
        start, end = args.startend
    else:
        start, end = 0, min(args.max_preds, num_preds)

    search = se_search.se_search

    if args.ne_thresh:
        threshes = OrderedDict()
        for i in range(0, len(args.ne_thresh), 4):
            threshes[args.ne_thresh[i]] = (int(args.ne_thresh[i+1]), int(args.ne_thresh[i+2]),
                                           int(args.ne_thresh[i+3])) # nmoves, nne, K

    with open(args.out_fi, "w+") as f:
        for ii in range(start, end, args.bsz):
            batchidxs = list(range(ii, min(ii+args.bsz, end)))
            nelist, neoffs = [], [0] # neoffs[b] contains idx of first neighbor for example b
            for idx in batchidxs:
                if args.ne_thresh:
                    nmoves, threshnne, K  = get_thresh_stuff(nedists[idx], threshes)
                    #threshnne = get_nne(nedists[idx], threshes)
                    nelists[idx] = nelists[idx][:nprotes+threshnne]
                else:
                    nmoves, K = args.max_moves, args.K
                nelist.extend(nelists[idx])
                neoffs.append(neoffs[-1] + len(nelists[idx]))

            if (ii+1) % args.log_interval == 0:
                print("predicting line", ii+1)
            batch_fin_hyps = search(batchidxs, nelist, neoffs, K, model, db, device,
                                    max_moves=nmoves, min_len=args.min_len,
                                    max_canvlen=args.max_canvlen, len_avg=(not args.no_len_avg),
                                    leftright=args.leftright, only_copy=args.only_copy)
            for b in range(len(batchidxs)):
                fin_hyps = batch_fin_hyps[b]
                if len(fin_hyps) == 0:
                    print("wtf2", batchidxs[b])
                    assert False
                argmax = torch.Tensor([hyp.score for hyp in fin_hyps]).argmax()
                if db.tokenizer is None:
                    pred = " ".join(fin_hyps[argmax.item()].canvas)
                else:
                    pred = ''.join(tok.replace('</w>', ' ') # should be equiv to tok's decode thing
                                   for tok in fin_hyps[argmax.item()].canvas).strip()
                f.write(pred)
                f.write("|||")
                f.write("{:.6f}".format(fin_hyps[argmax.item()].score))
                if args.get_trace:
                    trace = fin_hyps[argmax.item()].get_moves(db.val_srcs[batchidxs[b]], nelist, db)
                    tracestr = " ".join(["=>(%c) %s" % (srcc, " ".join(thing))
                                         for srcc, thing in trace])
                    f.write("|||")
                    f.write(tracestr)
                f.write("\n")

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', type=str, default="data/wb", help='datadir')
parser.add_argument("-tokfi", #default="data/giga/16k-bpe.tokenizer.json",
                    default=None, type=str, help="")
parser.add_argument("-split_dashes", action='store_true', help="")
parser.add_argument('-val_src_fi', type=str, default=None, help='if diff than in data/')
parser.add_argument('-get_trace', action='store_true', help='')
parser.add_argument('-leftright', action='store_true', help='')
parser.add_argument('-nne', type=int, default=200, help='only used to load stuff')
parser.add_argument("-prote_fi", default="", type=str, help="")
parser.add_argument('-bsz', type=int, default=4, help='batch size')
parser.add_argument('-seed', type=int, default=1111, help='random seed')
parser.add_argument('-cuda', action='store_true', help='use CUDA')
parser.add_argument('-log_interval', type=int, default=200, help='report interval')
parser.add_argument('-train_from', type=str, default='', help='')
parser.add_argument('-K', type=int, default=1, help='beam size')
parser.add_argument('-max_moves', type=int, default=23, help='') # 98% of wb < 23; 98% of iw < 32
parser.add_argument('-max_canvlen', type=int, default=200, help='')
parser.add_argument('-out_fi', type=str, default="preds.out", help='')
parser.add_argument('-val_nefi', type=str, default="val-nes.txt", help='')
parser.add_argument('-no_len_avg', action='store_true', help='')
parser.add_argument('-min_len', type=int, default=0, help='INCLUDES <tgt>, </tgt> (so maybe add 2)')
#parser.add_argument('-ne_thresh', type=int, default=0, help='')
parser.add_argument('-ne_thresh', nargs='+', type=float, default=None,
                    help='[min_score, nmoves, nne, K]')
parser.add_argument('-max_preds', type=int, default=4000, help='')
parser.add_argument('-only_copy', action='store_true', help='')
parser.add_argument('-restrict_nes', type=int, default=5, help='gross but whatever')
parser.add_argument('-debug', type=int, default=-1, help='')
parser.add_argument('-startend', nargs='+', type=int, default=None, help='')

if __name__ == "__main__":
    args = parser.parse_args()
    args.arbl = False
    print(args)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with -cuda")
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    assert args.train_from
    saved_stuff = torch.load(args.train_from)
    saved_args = saved_stuff["opt"]

    args.enclose, args.sel_firstlast_idxing = saved_args.enclose, saved_args.sel_firstlast_idxing
    args.vocopts = saved_args.vocopts
    print("enclose", args.enclose, "firstlast", args.sel_firstlast_idxing)
    db = data.ValDB(args) # protes should already be added in if
    assert db.sel_firstlast_idxing


    mod_ctor = mutils.BartThing
    if not hasattr(saved_args, "leftright"):
        saved_args.leftright = args.leftright
    model = mod_ctor(len(db.d), db.d.gen_voc_size, saved_args)
    model.load_state_dict(saved_stuff["sd"])
    model = model.to(device)
    restrict_fn = None

    if args.debug != -2:
        with torch.no_grad():
            batched_pred(db, model, device, args, restrict_fn=restrict_fn)
