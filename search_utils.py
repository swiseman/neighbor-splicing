import torch
from torch.nn.utils.rnn import pad_sequence

def init_search(batchidxs, nelist, db):
    bsz = len(batchidxs)
    if db.sel_firstlast_idxing:
        srcs = pad_sequence([torch.LongTensor(db.d.toks2idxs(db.val_srcs[idx]))
                             for idx in batchidxs], padding_value=db.pad_idx)
        assert srcs[0][0].item() != db.d.w2i["<src>"]
        neighbs = pad_sequence([torch.LongTensor(db.d.toks2idxs(db.train_tgts[idx]))
                                for idx in nelist], padding_value=db.pad_idx)
    else:
        padsrcs = [[db.d.w2i["<src>"]] + db.d.toks2idxs(db.val_srcs[idx])
                   for idx in batchidxs]
        [src.append(db.d.w2i["</src>"]) for src in padsrcs]
        srcs = pad_sequence([torch.LongTensor(padsrc) for padsrc in padsrcs],
                            padding_value=db.pad_idx)

        padtgts = [[db.d.w2i["<tgt>"]] + db.d.toks2idxs(db.train_tgts[idx])
                   for idx in nelist]
        [padtgt.append(db.d.w2i["</tgt>"]) for padtgt in padtgts]
        neighbs = pad_sequence([torch.LongTensor(padtgt) for padtgt in padtgts],
                               padding_value=db.pad_idx)

    canvases = torch.LongTensor(
        [db.d.w2i["<tgt>"], db.d.w2i["</tgt>"]]).view(-1, 1).repeat(1, bsz)
    relidxs = torch.LongTensor([0, 0]).view(-1, 1).repeat(1, bsz)
    lengths = torch.LongTensor([0, 0]).view(-1, 1).repeat(1, bsz)
    max_srclen = srcs.size(0)
    ufeats = torch.zeros(max_srclen, bsz, dtype=torch.long)
    for b in range(bsz):
        ufeats[len(db.val_srcs[batchidxs[b]]):, b].fill_(3) # padding
    inslocs = torch.LongTensor([0]).repeat(bsz)
    return srcs, ufeats, neighbs, canvases, relidxs, lengths, inslocs


def get_updated_canvs(hyps, db, device):
    canvases = pad_sequence([torch.LongTensor(db.d.toks2idxs(hyp.canvas))
                             for hyps_b in hyps for hyp in hyps_b], padding_value=db.pad_idx)
    relidxs = pad_sequence([torch.LongTensor(hyp.rellist)
                            for hyps_b in hyps for hyp in hyps_b], padding_value=-1)
    lengths = pad_sequence([torch.LongTensor(hyp.lengths)
                            for hyps_b in hyps for hyp in hyps_b], padding_value=db.pad_idx)
    ufeats = pad_sequence([hyp.ufeats for hyps_b in hyps for hyp in hyps_b], padding_value=3)
    canvases, relidxs, lengths = canvases.to(device), relidxs.to(device), lengths.to(device)
    ufeats = ufeats.to(device)
    return canvases, relidxs, lengths, ufeats


def make_nemask(neighbs, neoffs, pad_idx):
    """
    returns bsz x nelen*nne mask with 1s for pad tokens or a neighbor from a different batch idx
    """
    bsz = len(neoffs) - 1
    nelen, nne = neighbs.size()
    nemask = (neighbs.view(-1) == pad_idx).unsqueeze(0).expand(bsz, -1).contiguous()
    nmv = nemask.view(bsz, nelen, nne)
    for b in range(bsz):
        nmv[b, :, :neoffs[b]].fill_(1)
        nmv[b, :, neoffs[b+1]:].fill_(1)
    return nemask


def get_trace(moves, val_src, nelist, db):
    canv = ["<tgt>", "</tgt>"]
    trace = []
    for move in moves:
        ktype, tneidx, tl, tr, tj, tk = move
        srcc = None
        if ktype == 2:
            span = val_src[tl:tr]
            srcc = "s"
        elif ktype == 1:
            span = [db.d.i2w[tneidx]]
            srcc = "v"
        else:
            span = db.train_tgts[nelist[tneidx]][tl:tr]
            srcc = "n"
        canv = canv[:tj] + span + canv[tk:]
        trace.append((srcc, canv[:]))
    return trace

class Hyp(object):
    """
    hypotheses for beam search
    """
    def __init__(self, canvas, rellist, lengths, ufeats, move):
        assert canvas[0] == "<tgt>"
        self.canvas = canvas
        self.rellist = rellist
        self.lengths = lengths
        self.ufeats = ufeats
        self.curr_move = move # format is (ktype, tneidx, tl, tr, tj, tk)
        self.mvidx = 0
        self.score = 0
        self.parent = None

    def get_start_child(self, ktype, tneidx, tl, tj, score):
        move = (ktype, tneidx, tl, None, tj, None)
        # keep previous canvas for now
        nuhyp = Hyp(self.canvas, self.rellist, self.lengths, self.ufeats, move)
        nuhyp.score = score
        nuhyp.parent = self
        nuhyp.mvidx = self.mvidx + 1
        return nuhyp

    def get_end_child(self, db, ii, neidxs, predr, predk, score):
        ktype, tneidx, tl, _, tj, _ = self.curr_move
        mvidx = self.mvidx
        nuufeats = self.ufeats.clone()
        if ktype == 2: # from src
            span = db.val_srcs[ii][tl:predr]
            nuufeats[tl:predr].fill_(1)
        elif ktype == 1: # a word
            span = [db.d.i2w[tneidx]]
        else: # from a neighbor
            span = db.train_tgts[neidxs[tneidx]][tl:predr]
        nucanv = self.canvas[:tj] + span + self.canvas[predk:]
        nurels = self.rellist[:tj] + [mvidx]*len(span) + self.rellist[predk:]
        nulens = self.lengths[:tj] + [len(span)]*len(span) + self.lengths[predk:]
        nuhyp = Hyp(nucanv, nurels, nulens, nuufeats, (ktype, tneidx, tl, predr, tj, predk))
        nuhyp.mvidx, nuhyp.score, nuhyp.parent = mvidx, score, self
        return nuhyp

    def get_start_final_child(self, score, len_avg=True):
        nuhyp = Hyp(self.canvas, self.rellist, self.lengths, self.ufeats, None)
        nuhyp.parent = self
        nuhyp.mvidx = self.mvidx + 1
        nmoves = (2*nuhyp.mvidx - 1) if len_avg else 1
        nuhyp.score = score/nmoves
        return nuhyp

    def get_sel_child(self, db, ii, neidxs, ktype, predc, predl, predr, score):
        nuufeats = self.ufeats.clone()
        inc = int(not db.sel_firstlast_idxing)
        if ktype == 2: # from src
            span = db.val_srcs[ii][predl:predr]
            nuufeats[predl+inc:predr+inc].fill_(1)
        elif ktype == 1: # a word
            span = [db.d.i2w[predc]]
        else: # from a neighbor
            span = db.train_tgts[neidxs[predc]][predl:predr]
        _, _, _, _, jj, kk = self.curr_move
        mvidx = self.mvidx # mvidx of insert...
        nucanv = self.canvas[:jj] + span + self.canvas[kk:]
        nurels = self.rellist[:jj] + [mvidx+1]*len(span) + self.rellist[kk:]
        nulens = self.lengths[:jj] + [len(span)]*len(span) + self.lengths[kk:]
        numove = (ktype, predc, predl, predr, jj, kk)
        nuhyp = Hyp(nucanv, nurels, nulens, nuufeats, numove)
        nuhyp.mvidx, nuhyp.score, nuhyp.parent = mvidx, score, self
        return nuhyp

    def get_ins_child(self, jj, kk, score):
        numove = (None, None, None, None, jj, kk)
        nuhyp = Hyp(self.canvas, self.rellist, self.lengths, self.ufeats, numove)
        nuhyp.score = score
        nuhyp.parent = self
        nuhyp.mvidx = self.mvidx + 1
        return nuhyp

    def get_catbl_ins_child(self):
        # self.next_insert = (len(self.canvas)-1, len(self.canvas)-1)
        return self

    def get_ins_final_child(self, score, len_avg=True):
        nuhyp = Hyp(self.canvas, self.rellist, self.lengths, self.ufeats, None)
        nmoves = 2*(self.mvidx + 1) if len_avg else 1 # include this ins but not first ins in count
        nuhyp.score = score/nmoves
        nuhyp.parent = self
        return nuhyp

    def get_moves(self, val_src, nelist, db):
        """
        does the full trace if val_src is not None
        """
        moves = []
        if self.curr_move is None: # a properly ended hypothesis
            curr = self.parent
        else: # didn't predict an end
            curr = self
        while curr is not None and curr.curr_move is not None:
            moves.append(curr.curr_move) # should be complete
            curr = curr.parent.parent # go to previous full move
        moves = moves[::-1]
        if val_src is not None:
            return get_trace(moves, val_src, nelist, db)
        return moves
