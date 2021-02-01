from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

import cky

############################ Forming Canvases ###################################
def get_move_canvases(moves_list, srcs_list, idx, srclen, next_state, boc, eoc,
                      max_canvlen=10000, ignore_unk=False):
    """
    srclen shouldbe >= length of src (if batching longer might be good)
    if max_canvlen is set we keep doing moves until things aren't that long...
    """
    # netgts is a list of lists b/c for trees we want multiple tgts
    canvases, netgts = [[boc, eoc]], [[moves_list[idx][0]]]
    mvidxs, srcfeats = [[0, 0]], [torch.LongTensor(srclen).zero_()]
    lastcanv, lastrelidx, lastfeats = canvases[0], mvidxs[0], srcfeats[0]
    # mvidx = 0 # we'll use 1-based features tho...
    for m in range(len(moves_list[idx])):
        move = moves_list[idx][m]
        canvas, relidx, nufeats = next_state(
            lastcanv, lastrelidx, lastfeats, move, m, srcs_list[idx])
        if len(canvas) <= max_canvlen and (not ignore_unk or m + 1 == len(moves_list[idx])
                                           or moves_list[idx][m+1][0] != -1): # hack assuming 1
            canvases.append(canvas)
            mvidxs.append(relidx)
            srcfeats.append(nufeats)

            if m < len(moves_list[idx]) - 1:
                netgts.append([moves_list[idx][m+1]])

        lastcanv, lastrelidx, lastfeats = canvas, relidx, nufeats

    # add final tgt
    if len(lastcanv) <= max_canvlen:
        netgts.append([(None, 0, 1, 0, 0)])

    assert len(canvases) == len(netgts)
    assert len(canvases) == len(mvidxs)
    assert len(canvases) == len(srcfeats)
    return canvases, netgts, mvidxs, srcfeats


def get_tree_canvases(trees_list, srcs_list, tidx, srclen, next_state, boc, eoc,
                      max_canvlen=10000, debug=False):
    """
    srclen shouldbe >= length of src (if batching longer might be good)
    if max_canvlen is set we keep doing moves until things aren't that long...
    """
    canvases, netgts = [[boc, eoc]], []
    mvidxs, srcfeats = [[0, 0]], [torch.LongTensor(srclen).zero_()]
    src = srcs_list[tidx]
    lastcanv, lastrelidx, lastfeat = canvases[0], mvidxs[0], srcfeats[0]

    # put top-level children and an order for them on the stack
    stack, permstack = [trees_list[tidx]], [None]
     # get a stack of moves we can manipulate
    movestack, offstack = [None], [(0, 0)] # stores (sum of offsets to left, current offset)
    mvidx = 0
    while stack:
        top = stack.pop() # last thing we put on; should correspond to current canvas
        par_offsum, _ = offstack.pop() #
        # get tgts for current canvas
        if cky.is_leaf(top): # the tgts are like the previous things
            if not movestack[-1]: # we're done
                break

            tgts = [tuple(move) for move in movestack[-1]]
            # get next random move
            next_move = movestack[-1].pop() # were randomly ordered before

            if movestack[-1]: # if still more targets left update them
                roffset = offstack[-1][1] # offset of thing we're about to make a canvas of
                corspnding_ridx = permstack[-1][-1] # ridx corresponding to next_move
                permstack[-1] = permstack[-1][:-1]
                # update remaining moves on movestack and on offstack
                for i, idx in enumerate(permstack[-1]):
                    if idx > corspnding_ridx:
                        movestack[-1][i][3] += roffset
                        offsumi, curoffi = offstack[-len(permstack[-1])-1+i]
                        offstack[-len(permstack[-1])-1+i] = (offsumi - roffset, curoffi)
            else: # we're done with this subtree
                movestack.pop()
                permstack.pop()
        else: # has children
            assert top[0][0] in ['X', 'S']
            tgts, offsets, leftoffset = [], [], 0 # left offset just for this subtree
            for i in range(len(top[1])):
                cnode = cky.get_node(top[1][i])
                _, neidx, l, r, j, skip, size = cnode
                j -= (leftoffset + par_offsum)
                tgts.append((neidx, l, r, j, skip))
                # keep track of this child's contribution to offset from left;
                # this tells us how much we can shift NEXT child to the left
                offset_i = size - skip # N.B. skip should only be > 0 if top[0][0] == 'X'
                offsets.append((par_offsum+leftoffset, offset_i))
                leftoffset += offset_i
            # get next (random) move to make next canvas
            perm = torch.randperm(len(top[1]))
            if debug:
                perm = torch.tensor(range(len(top[1]))[::-1])
            # put children on the stack
            stack.extend([top[1][idx] for idx in perm])
            # make a canvas from last child on stack that we're about to pop
            ridx = perm[-1].item()
            next_move = tgts[ridx]
            if len(top[1]) > 1: # save remaining child moves to be used later
                perm = perm[:-1]
                roffset = offsets[ridx][1]
                remmoves = []
                for idx in perm:
                    remmoves.append(list(tgts[idx]))
                    if idx < ridx: # left of ridx; don't need to change anything
                        offstack.append(offsets[idx])
                    else: # right of ridx; need to update move's j and corresponding offset
                        remmoves[-1][3] += roffset
                        # also have to correct offset
                        offstack.append((offsets[idx][0]-roffset, offsets[idx][1]))

                movestack.append(remmoves)
                permstack.append(perm)

            # add offset we're about to pop (just like for regular stack)
            offstack.append(offsets[ridx])
        netgts.append(tgts)

        # make canvas with the randomly chosen move
        canv, relidx, srcfeat = next_state(
            lastcanv, lastrelidx, lastfeat, next_move, mvidx, src)
        canvases.append(canv)
        mvidxs.append(relidx)
        srcfeats.append(srcfeat)
        lastcanv, lastrelidx, lastfeat = canv, relidx, srcfeat
        mvidx += 1
    # we're done and last tgt is to stop
    netgts.append([(None, 0, 1, 0, 0)])
    assert len(canvases) == len(netgts)
    assert len(canvases) == len(mvidxs)
    assert len(canvases) == len(srcfeats)
    return canvases, netgts, mvidxs, srcfeats

############################ Getting Targets #####################################

def get_starttgt_matches(idx, rultgt, tgt_span, uni_locs, neidxs, nne,
                         nelen, src, offset, train_tgts, vocsize, unkidx,
                         val=False):
    matches, splen = [], len(tgt_span)
    if uni_locs is not None:
        for kk, start in uni_locs[tgt_span[0]]:
            if ((val or neidxs[kk] != idx) # ignore neighbor and also orig tgt idx
                    and train_tgts[neidxs[kk]][start:start+splen] == tgt_span
                    and offset + start*nne + kk != rultgt):
                matches.append(offset+start*nne + kk)
    # also check in src
    for j in range(len(src)-splen+1):
        stgt = offset + nelen*nne + vocsize + j
        if stgt != rultgt and src[j:j+splen] == tgt_span:
            matches.append(stgt)
    ## also if it's just a word
    #if splen == 1 and tgt_span[0] != unkidx and tgt_span[0] < vocsize:
    #    matches.append(offset + nelen*nne + tgt_span[0])
    return matches

def get_startend_tgts(alnetgts, nelen, nne, fin_idx, vocsize, srclen, batch, canv2ii,
                      train_tgts, ne2neidx, uni_locs, neidxs, srcs_list, unkidx, val,
                      leftright=False):
    starttgts = []
    colpercanv = nelen*nne+vocsize+srclen
    ralnetgts = []
    for jj in range(len(alnetgts)):
        bidx = batch[canv2ii[jj]]
        starttgt = []
        alnetgts_jj = alnetgts[jj] # this list is only longer than 1 if we don't flatten
        # pick a random tgt to calculate the endtgt from (TODO: consider all?)
        rtgt_idx = torch.randint(len(alnetgts_jj), (1,)).item()
        for tt, (tneidx, tl, tr, tj, tskip) in enumerate(alnetgts_jj):
            # offset really uses tj+1-1 bc of prepended <tgt> and then fenceposting
            offset = tj*colpercanv if not leftright else 0
            # get tgt span and other matching spans among neighbors if any
            if tneidx is None:
                ktype, trulen = 1, 1
                starttgt.append(fin_idx)
                tneidx = 0
                if leftright: # use end of last insert as usual
                    prevtgt = ralnetgts[-1] # must be a previous one
                    tj = prevtgt[4] + prevtgt[3] - prevtgt[2]
            elif tneidx >= 0: # from a neighbor
                ktype, trulen = 0, len(train_tgts[tneidx])
                tgt_span = train_tgts[tneidx][tl:tr]
                starttgt.append(offset+tl*nne+ne2neidx[tneidx])
                starttgt.extend(get_starttgt_matches( # other matches in src/nes/voc
                    bidx, starttgt[-1], tgt_span, uni_locs, neidxs, nne, nelen, srcs_list[bidx],
                    offset, train_tgts, vocsize, unkidx, val=val))
                tneidx = ne2neidx[tneidx]
            elif tneidx == -2: # from src
                ktype, trulen = 2, len(srcs_list[bidx])
                tgt_span = srcs_list[bidx][tl:tr]
                starttgt.append(offset+nelen*nne+vocsize+tl)
                starttgt.extend(get_starttgt_matches( # other matches just in src
                    bidx, starttgt[-1], tgt_span, None, neidxs, nne, nelen, srcs_list[bidx],
                    offset, train_tgts, vocsize, unkidx, val=val))
                tneidx = jj
            else: # a word; don't think we get here if it's also among neighbors...
                ktype, trulen = 1, 1
                starttgt.append(offset+nelen*nne - tneidx) # tneidx is negative
                tneidx = -tneidx
            if tt == rtgt_idx:
                #alnetgts[jj] = (ktype, tneidx, tl, tr, tj, tj+tskip, trulen)
                ralnetgts.append((ktype, tneidx, tl, tr, tj, tj+tskip, trulen))
        starttgts.append(torch.LongTensor(starttgt))
    return starttgts, ralnetgts
