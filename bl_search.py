import argparse
import time

from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import db2 as data
import model_utils as mutils
from search import get_thresh_stuff

class Hyp(object):
    def __init__(self, hyp, score):
        self.hyp = hyp
        self.score = score


def aggregate_logprobs(lps, per_batch_idxs, K):
    """
    lps - bsz*K x nelen*nne + ngentypes + srcpairs
    per_batch_idxs - bsz length list of lists of idxs
    """
    # these need to map src stuff to the first idx in idxlist if necessary
    # should contain at most 1 idx among the neighbor spans
    for b, b_idxs in enumerate(per_batch_idxs):
        for idxs in b_idxs:
            aglp = lps[b*K:(b+1)*K].index_select(1, idxs).logsumexp(1) # K
            lps[b*K:(b+1)*K, idxs[0].item()].copy_(aglp)
            lps[b*K:(b+1)*K].index_fill_(1, idxs[1:], -float("inf"))


def get_repeat_info(db, nelist, batchidxs, max_nelen, neoffs, device):
    nne = len(nelist) if nelist is not None else 0
    per_batch_idxs = []
    for b in range(len(batchidxs)):
        t2i_b = defaultdict(list)
        # account for neighbor and src tokens
        [t2i_b[toke].append(max_nelen*nne + db.d.gen_voc_size + j)
         for j, toke in enumerate(db.val_srcs[batchidxs[b]])]
        if nelist is not None:
            [t2i_b[toke].append(w*nne + n + neoffs[b])
             for n, neidx in enumerate(nelist[neoffs[b]:neoffs[b+1]])
             for w, toke in enumerate(db.train_tgts[neidx])]

        # add regular tgt vocab stuff too
        for tok in t2i_b.keys():
            if tok in db.d.w2i and db.d.w2i[tok] < db.d.gen_voc_size:
                t2i_b[tok].append(max_nelen*nne + db.d.w2i[tok])
        per_batch_idxs.append([torch.LongTensor(locs).to(device)
                               for locs in t2i_b.values() if len(locs) > 1])

    return per_batch_idxs


def pred2token(pred, db, bidx, nelist, max_nelen):
    nne = len(nelist) if nelist is not None else 0
    if pred < max_nelen*nne: # from a neighbor
        row, col = pred // nne, pred % nne
        toke = db.train_tgts[nelist[col]][row]
    elif pred < max_nelen*nne + db.d.gen_voc_size: # a word
        toke = db.d.i2w[pred - max_nelen*nne]
    else: # from src
        toke = db.val_srcs[bidx][pred - max_nelen*nne - db.d.gen_voc_size]
    return toke


def search(batchidxs, nelist, neoffs, K, model, db, device, max_moves=50, min_len=0,
           len_avg=True, only_copy=False):
    model.eval()
    bsz = len(batchidxs)
    fin_hyps = [[] for _ in range(bsz)]
    hyps = [[Hyp(["<bos>"], 0)] for _ in range(bsz)]

    srcs = pad_sequence( # srclen x bsz
        [torch.LongTensor(db.d.toks2idxs(db.val_srcs[bidx]))
         for bidx in batchidxs], padding_value=db.pad_idx)
    inps = torch.LongTensor(1, bsz).fill_(db.d.w2i["<bos>"])
    if nelist is not None:
        neighbs = pad_sequence( # nelen x nne
            [torch.LongTensor(db.d.toks2idxs(db.train_tgts[neidx]))
             for neidx in nelist], padding_value=db.pad_idx)
        nelen, nne = neighbs.size()
    else:
        neighbs = torch.LongTensor()
        nelen, nne = 0, 0

    srcs, neighbs, inps = srcs.to(device), neighbs.to(device), inps.to(device)

    encsrc, srcmask = model.enc_fwd(model.encoder, srcs, None, db.pad_idx, neenc=False)
    if nelist is not None:
        encne = model.ne_encode(neighbs, db.pad_idx)
        skeys = model.Slin(encsrc) # srclen x bsz x dim
        nkeys = model.Nlin(encne.view(-1, encne.size(2))) # nelen*nne x dim
        wkeys = model.Wlin(model.lut.weight[:model.ngentypes]) # negentypes x dim

        nemask = (neighbs.view(-1) == db.pad_idx).unsqueeze(0) # 1 x nelen*nne
    else:
        skeys = model.Slin(encsrc)
        wkeys = model.lut.weight[:model.ngentypes]

    mvidx = 0
    remaining = set(range(bsz))
    scores = torch.zeros(bsz, 1).to(device)

    repidxs_per_batch = get_repeat_info(db, nelist, batchidxs, nelen, neoffs, device)

    while len(remaining) > 0 and mvidx < max_moves:
        _, nhyps = inps.size()
        # just need last timestep
        enctgt = model.dec_fwd(inps, db.pad_idx, encsrc, srcmask)[-1] # nhyps x dim
        if nelist is not None:
            C = model.Clin(enctgt) # nhyps x dim
            nscores = C.mm(nkeys.t()) # nhyps x nelen*nne
            nscores = nscores.masked_fill(nemask.expand(nhyps, nscores.size(1)), -float("inf"))
            sscores = C.unsqueeze(1).bmm(skeys.permute(1, 2, 0)).squeeze(1) # nhyps x srclen
            sscores = sscores.masked_fill(srcmask, -float("inf"))
            wscores = C.mm(wkeys.t()) # nhyps x V
            all_scores = torch.cat([nscores, wscores, sscores], 1)
        else:
            sscores = enctgt.unsqueeze(1).bmm(skeys.permute(1, 2, 0)).squeeze(1) # nhyps x srclen
            sscores = sscores.masked_fill(srcmask, -float("inf"))
            wscores = enctgt.mm(wkeys.t())
            all_scores = torch.cat([wscores, sscores], 1)

        logprobs = F.log_softmax(all_scores, dim=1) # nhyps x nelen*nne + V + srclen
        aggregate_logprobs(logprobs, repidxs_per_batch, K)
        logprobs.add_(scores.view(-1, 1))
        logprobs[:, nelen*nne+db.d.w2i["<unk>"]].fill_(-float("inf")) # disallow unks
        if only_copy:
            logprobs[:, nelen*nne+db.d.nspecial:nelen*nne+db.d.gen_voc_size].fill_(-float("inf"))

        maxes, argmaxes = logprobs.view(bsz, -1).topk(2*K, dim=1) # bsz x 2K
        nuhyps, paridxs, predtoks = [], [], []
        for b in range(bsz):
            if b not in remaining:
                nuhyps.append(hyps[b])
                continue
            hyps_b = []
            for k in range(2*K): # just heuristic...
                parent = argmaxes[b, k].item() // logprobs.size(1)
                pred = argmaxes[b, k].item() % logprobs.size(1)
                predtok = pred2token(pred, db, batchidxs[b], nelist, nelen)
                nuhyp = Hyp(hyps[b][parent].hyp + [predtok], maxes[b, k].item())
                # if predtok == "<eos>" and k == 0: # we're done
                #     fin_hyps[b].append(nuhyp)
                #     remaining.remove(b)
                #     hyps_b = [Hyp(nuhyp.hyp, -1e38) for _ in range(K)]
                #     break
                if predtok == "<eos>" and len(nuhyp.hyp) > min_len: # has an extra eos...
                    fin_hyps[b].append(nuhyp)
                    if len_avg:
                        nuhyp.score = nuhyp.score/len(nuhyp.hyp)
                else:
                    if k != len(hyps_b):
                        assert k > len(hyps_b)
                        maxes[b, len(hyps_b)] = maxes[b, k]
                    hyps_b.append(nuhyp)
                    paridxs.append(b*(nhyps//bsz) + parent)
                    predtoks.append(predtok)
                if len(hyps_b) == K:
                    break
            if len(hyps_b) < K:
                print("couldn't find enough short hyps for", batchidxs[b])
                # will fail if no hyps_b[0]
                [hyps_b.append(hyps_b[0]) for _ in range(K - len(hyps_b))]
                paridxs.extend([paridxs[-1]]*(K - len(hyps_b)))
                predtok.extend([db.d.w2i["<pad>"]]*(K - len(hyps_b)))
            nuhyps.append(hyps_b)

        hyps = nuhyps
        scores = maxes[:, :K].contiguous()
        nhyps = bsz*K
        paridxs = torch.LongTensor(paridxs).to(device)
        assert paridxs.size(0) == nhyps
        assert len(predtoks) == nhyps

        mvidx += 1
        repinps = inps.index_select(1, paridxs) # prevlength x nhyps
        curr_wrds = torch.LongTensor([db.d.add_word(wrd) for wrd in predtoks]).to(device)
        inps = torch.cat([repinps, curr_wrds.view(1, -1)], 0)
        encsrc = encsrc.index_select(1, paridxs) # srclength x nhyps x dim
        skeys = skeys.index_select(1, paridxs) # srclength x nhyps x dim
        srcmask = srcmask[paridxs] # nhyps x srclength

    # ok so now we see if we have any hypotheses that haven't finished and take the best
    for b in range(bsz):
        if not fin_hyps[b]:
            print("didn't finish!")
            best_score, best_hyp = -float("inf"), None
            for hyp in hyps[b]:
                if hyp.score > best_score:
                    best_score, best_hyp = hyp.score, hyp
            fin_hyps[b].append(best_hyp)

    return fin_hyps

def batched_pred(db, model, device, args, restrict_fn=None):
    if restrict_fn is not None:
        for ii in range(len(db.val_neidxs)):
            restricted_nes = restrict_fn(db.val_neidxs[ii], db, ii)
            db.val_neidxs[ii] = restricted_nes[:args.restrict_nes]
            if len(db.val_neidxs[ii]) < args.restrict_nes:
                print("not enough restricted neighbors for", ii)

    nelists, nedists, nprotes = db.val_neidxs, db.val_ne_dists, len(db.protes)
    model.eval()
    num_preds = len(db.val_srcs)
    print("num_preds", num_preds, len(nelists))
    print("nprotes", nprotes)
    assert len(nelists) == num_preds

    if args.debug >= 0:
        start, end = args.debug, args.debug+1
    elif args.startend:
        start, end = args.startend
    else:
        start, end = 0, min(args.max_preds, num_preds)

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
            if args.no_ne:
                nelist = None
            batch_fin_hyps = search(batchidxs, nelist, neoffs, K, model, db, device,
                                    max_moves=nmoves, min_len=args.min_len,
                                    len_avg=(not args.no_len_avg), only_copy=args.only_copy)
            for b in range(len(batchidxs)):
                fin_hyps = batch_fin_hyps[b]
                if len(fin_hyps) == 0:
                    print("wtf2", batchidxs[b])
                    assert False
                argmax = torch.Tensor([hyp.score for hyp in fin_hyps]).argmax()
                if db.tokenizer is None:
                    pred = " ".join(fin_hyps[argmax.item()].hyp)
                else:
                    pred = ''.join(tok.replace('</w>', ' ') # should be equiv to tok's decode thing
                                   for tok in fin_hyps[argmax.item()].hyp).strip()
                f.write(pred)
                f.write("|||")
                f.write("{:.6f}".format(fin_hyps[argmax.item()].score))
                f.write("\n")

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', type=str, default="data/wb", help='datadir')
parser.add_argument("-tokfi", #default="data/giga/16k-bpe.tokenizer.json",
                    default=None, type=str, help="")
parser.add_argument("-split_dashes", action='store_true', help="")
parser.add_argument('-val_src_fi', type=str, default=None, help='if diff than in data/')
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
parser.add_argument('-ne_thresh', nargs='+', type=float, default=None,
                    help='[min_score, nmoves, nne, K]')
parser.add_argument('-max_preds', type=int, default=4000, help='')
parser.add_argument('-only_copy', action='store_true', help='')
parser.add_argument('-restrict_nes', type=int, default=5, help='gross but whatever')
parser.add_argument('-debug', type=int, default=-1, help='')
parser.add_argument('-startend', nargs='+', type=int, default=None, help='')
parser.add_argument('-no_ne', action='store_true', help='')

if __name__ == "__main__":
    args = parser.parse_args()
    args.arbl = True
    args.leftright = False
    print(args)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with -cuda")
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    assert args.train_from
    saved_stuff = torch.load(args.train_from)
    saved_args = saved_stuff["opt"]
    saved_args.leftright = False # just a reverse compat thing
    args.enclose, args.sel_firstlast_idxing = saved_args.enclose, saved_args.sel_firstlast_idxing
    args.vocopts = saved_args.vocopts

    db = data.ValDB(args)

    model = mutils.TokenCopyBart(len(db.d), db.d.gen_voc_size, saved_args)
    model.load_state_dict(saved_stuff["sd"])
    model = model.to(device)
    restrict_fn = None

    with torch.no_grad():
        tic = time.perf_counter()
        batched_pred(db, model, device, args, restrict_fn=restrict_fn)
        print("took", time.perf_counter() - tic)
