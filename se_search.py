import torch

import model_utils as mutils
import search_utils as sutils

def endidx2pred(endidx, max_remlen):
    tk = endidx // max_remlen # don't decrement b/c assuming prepended <tgt>
    tr = endidx % max_remlen
    tr += 1 # increment to undo firstlast idxing
    return tr, tk


def startidx2pred(startidx, colpercanv, nelen, nne, nelist, db):
    tj, col = startidx // colpercanv, startidx % colpercanv
    tj += 1 # because we fenceposted; not undoing increment by prepended <tgt>
    ktype = None
    if col < nelen*nne: # from a neighbor
        tl, neidx = col // nne, col % nne
        ktype, trulen = 0, len(db.train_tgts[nelist[neidx]])
    elif col < nelen*nne + db.d.gen_voc_size: # from vocab
        tl, neidx = 0, col - nelen*nne
        ktype, trulen = 1, 1
    else: # from src
        tl, neidx = col - nelen*nne - db.d.gen_voc_size, None
        ktype, trulen = 2, None
    return ktype, neidx, tl, tj, trulen


def se_search(batchidxs, nelist, neoffs, K, model, db, device, max_moves=25,
              min_len=0, max_canvlen=200, len_avg=True, leftright=False, only_copy=False):
    model.eval()
    bsz = len(batchidxs)
    fin_hyps = [[] for _ in range(bsz)]
    seen_canvs = [set() for _ in range(bsz)]

    # get initial conditions by rolling in to zero
    srcs, ufeats, neighbs, canvases, relidxs, _, _ = sutils.init_search(
        batchidxs, nelist, db)
    max_nelen = 200 # def an overestimate
    emask = torch.ones(bsz*K, max_canvlen, max_nelen, dtype=torch.bool).to(device)
    max_srclen = srcs.size(0)

    # book keeping: hyps are bsz x K list of (canvas, rellist, lengths, ufeats, next_insert)
    hyps = [[sutils.Hyp(["<tgt>", "</tgt>"], [0, 0], [0, 0],
                        torch.zeros(max_srclen, dtype=torch.long), None)]
            for _ in range(bsz)]

    srcs, ufeats, neighbs = srcs.to(device), ufeats.to(device), neighbs.to(device)
    canvases, relidxs = canvases.to(device), relidxs.to(device)

    encsrc, enccanv, _ = model.src_encode(
        srcs, ufeats, None, canvases, relidxs, db.pad_idx)
    encne = model.ne_encode(neighbs, db.pad_idx)

    nemask = sutils.make_nemask(neighbs, neoffs, db.pad_idx)
    canvmask = canvases == db.pad_idx # canvlen x bsz

    mvidx = 0
    remaining = set(range(bsz))
    scores = torch.zeros(bsz, 1).to(device)

    if leftright:
        enccanv = enccanv[0] # bsz x dim; corresponding to <tgt> as left fencepost

    #import ipdb; ipdb.set_trace()
    while len(remaining) > 0:
        canvlen = 1 if leftright else canvases.size(0)
        colpercanv = neighbs.nelement() + db.d.gen_voc_size + max_srclen

        canvmask = canvases == db.pad_idx # canvlen x bsz
        startlps = model.actmodel.get_start_lps( # bsz x canvlen*(nelen*nne+V+srclen)
            enccanv, canvmask, encne, nemask, encsrc, srcs, model.lut, pad_idx=db.pad_idx)
        assert startlps.size(1) == canvlen*colpercanv
        startlps.add_(scores.view(-1, 1))
        nnetoks = neighbs.nelement()
        startlps.view( # disallow unk
            startlps.size(0), canvlen, -1)[:, :, nnetoks + db.d.w2i["<unk>"]].fill_(-float("inf"))
        if only_copy: # can't invent any words
            startlps.view(
                startlps.size(0), canvlen, -1)[:, :, nnetoks+db.d.nspecial:nnetoks+db.d.gen_voc_size].fill_(
                    -float("inf"))

        # ideally we'd consider all the ways of extending our current hypotheses,
        # but that's probably too big. Instead we'll take each topk
        maxes, argmaxes = startlps.view(bsz, -1).topk(2*K, dim=1) # bsz x 2K
        nuhyps, netgts, par_idxs = [], [], []
        for b in range(bsz):
            nuhyps_b, currK = [], len(hyps[b])
            for k in range(2*K):
                parent = argmaxes[b, k].item() // startlps.size(1)
                startidx = argmaxes[b, k].item() % startlps.size(1)
                ktype, predne, predl, predj, trulen = startidx2pred(
                    startidx, colpercanv, neighbs.size(0), neighbs.size(1), nelist, db)
                if ktype == 2: # fix up src stuff
                    predne, trulen = b*currK + parent, len(db.val_srcs[batchidxs[b]])

                if ktype == 1 and (predj == 1 or leftright) and predne == db.d.w2i["<src>"]:
                    if (len(hyps[b][parent].canvas) >= min_len
                            and "<mask>" not in hyps[b][parent].canvas):
                        fin_hyps[b].append(hyps[b][parent].get_start_final_child(
                            maxes[b, k].item(), len_avg=len_avg))
                else:
                    if leftright: # we already know j, it's just what we're up to after last insert
                        parmove = hyps[b][parent].curr_move
                        if parmove is None: # first prediction
                            predj = 1
                        else:
                            predj = parmove[4] + parmove[3] - parmove[2] # par_tj + par_tr - par_tl
                    # netgt format is ktype, tneidx, tl, tr, tj, tk, trulen.
                    # predj-1 b/c db.get_endstuff() assumes it's predj w/o prepended <tgt>
                    netgts.append((ktype, predne, predl, 0, predj-1, 0, trulen))
                    if k != len(nuhyps_b): # must've skipped one
                        assert k > len(nuhyps_b)
                        maxes[b, len(nuhyps_b)] = maxes[b, k]
                    par_idxs.append(b*currK + parent)
                    nuhyps_b.append(hyps[b][parent].get_start_child(
                        ktype, predne, predl, predj, maxes[b, k].item()))
                if len(nuhyps_b) == K:
                    break
            assert len(nuhyps_b) == K
            nuhyps.append(nuhyps_b)
        assert len(netgts) == bsz*K

        hyps = nuhyps

        scores = maxes[:, :K].contiguous()
        remembs = model.actmodel.get_end_embs(encne, encsrc, model.lut, netgts) # maxlen x bsz x dim

        if leftright:
            endmask = emask[:len(netgts), 0, :remembs.size(0)]
            endmask.fill_(True)
            _ = mutils.get_leftright_endstuff(netgts, endmask) # fills endmask
            # update canvases; note max_canvlen may have changed but padding should work
            enccanv = enccanv.index_select( # bsz*K x dim
                0, torch.LongTensor(par_idxs).to(device))
        else:
            endmask = emask[:len(netgts), :canvases.size(0), :remembs.size(0)]
            endmask.fill_(True)
            _ = mutils.get_endstuff(netgts, endmask) # fills endmask; hacky
            # update canvases; note max_canvlen may have changed but padding should work
            enccanv = enccanv.index_select( # max_canvlen x bsz*K x dim
                1, torch.LongTensor(par_idxs).to(device))

        endlps = model.actmodel.get_end_lps1(enccanv, remembs, endmask) # bsz*K x canvlen*maxremlen
        # can get nans if we had a last-position start on the beam
        endlps[endlps.isnan()] = -float("inf")
        assert endlps.size(1) == canvlen*remembs.size(0)
        #print(netgts)
        #print(endlps.view(bsz*K, canvases.size(0), remembs.size(0)))
        endlps.add_(scores.view(-1, 1))

        maxes, argmaxes = endlps.view(bsz, -1).topk( # for leftright often < 2K
            min(2*K, endlps.nelement()//bsz), dim=1)
        nuhyps = []
        for b in range(bsz):
            nuhyps_b = []
            for k in range(argmaxes.size(1)):
                if torch.isinf(maxes[b, k]) and maxes[b, k] < 0: # beam too big and only illegal options left
                    break
                parent = argmaxes[b, k].item() // endlps.size(1)
                endidx = argmaxes[b, k].item() % endlps.size(1)
                predr, predk = endidx2pred(endidx, remembs.size(0))
                if leftright:
                    predk = hyps[b][parent].curr_move[4] # set to parent's tj
                nuhyp = hyps[b][parent].get_end_child(
                    db, batchidxs[b], nelist, predr, predk, maxes[b, k].item())
                canvkey = tuple(nuhyp.canvas)
                if len(canvkey) <= max_canvlen and canvkey not in seen_canvs[b]:
                    if k != len(nuhyps_b):
                        assert k > len(nuhyps_b)
                        maxes[b, len(nuhyps_b)] = maxes[b, k]
                    nuhyps_b.append(nuhyp)
                    seen_canvs[b].add(canvkey)
                if len(nuhyps_b) == K:
                    break
            if len(nuhyps_b) != K and len(nuhyps_b) > 0: # can happen if keep repeating
                nuhyps_b.extend([nuhyps_b[0]]*(K-len(nuhyps_b)))
            elif len(nuhyps_b) != K: # nothing short enough to add here, so we gotta finish
                nuhyps_b = hyps[b]
                mvidx = 999999999999999
            assert len(nuhyps_b) == K
            nuhyps.append(nuhyps_b)

        hyps = nuhyps
        scores = maxes[:, :K].contiguous()
        mvidx += 1
        if mvidx >= max_moves:
            break

        canvases, relidxs, _, ufeats = sutils.get_updated_canvs(hyps, db, device)

        if srcs.size(1) < canvases.size(1):
            assert srcs.size(1) == bsz
            srcs = srcs.unsqueeze(2).expand(-1, bsz, K).contiguous().view(srcs.size(0), bsz*K)
            nemask = nemask.unsqueeze(1).expand(bsz, K, -1).contiguous().view(bsz*K, nemask.size(1))

        encsrc, enccanv, _ = model.src_encode(srcs, ufeats, None, canvases, relidxs, db.pad_idx)

        if leftright: # select known nu left idx
            nuleftidxs = torch.LongTensor( # add segment length to tj then -1 for fenceposting
                [hyp.curr_move[4] + hyp.curr_move[3] - hyp.curr_move[2] - 1
                 for hyps_b in hyps for hyp in hyps_b]).to(device)
            nhyps = nuleftidxs.size(0)
            enccanv = enccanv.gather( # bsz x dim
                0, nuleftidxs.view(1, nhyps, 1).expand(1, nhyps, enccanv.size(2))).squeeze(0)

    # ok so now we see if we have any hypotheses that haven't finished and take the best
    for b in range(bsz):
        if not fin_hyps[b]:
            print("didn't finish!")
            best_score, best_hyp = -float("inf"), None
            for hyp in hyps[b]:
                if hyp.score > best_score: # all the same length, so don't need to avg
                    best_score, best_hyp = hyp.score, hyp
            fin_hyps[b].append(best_hyp)

    return fin_hyps
