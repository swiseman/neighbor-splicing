import math
from typing import Optional, Any
from torch import Tensor
import torch
import torch.nn as nn
#from torch.nn.functional import linear, softmax, dropout
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers.modeling_bart import BartEncoder, BartDecoder, SinusoidalPositionalEmbedding
from transformers.modeling_bart import DecoderLayer as BartDecoderLayer

def get_endstuff(netgts, endmask):
    """
    endmask - bsz x canvlen x max_ne_or_srclen, initialized with Trues
    used with first/last aligning decompositions
    """
    max_remlen = endmask.size(2)
    endttgts = torch.LongTensor(len(netgts))
    for b, netgt in enumerate(netgts):
        ktype, tneidx, tl, tr, tj, tk, trulen = netgt
        endttgts[b] = (tk+1)*max_remlen + tr - 1 # tk+1 bc <tgt>; tr-1 bc firstlast
        # only allow canv idxs after tj+1-1 and ne ends starting at tl
        endmask[b, tj+1:, tl:trulen].fill_(False)
    return endttgts

def get_leftright_endstuff(netgts, endmask):
    """
    endmask - bsz x max_ne_or_srclen, initialized with Trues
    """
    max_remlen = endmask.size(1)
    endttgts = torch.LongTensor(len(netgts))
    for b, netgt in enumerate(netgts):
        ktype, tneidx, tl, tr, tj, tk, trulen = netgt
        endttgts[b] = tr - 1 # tr-1 bc firstlast
        # only allow ne ends starting at tl
        endmask[b, tl:trulen].fill_(False)
    return endttgts

def neg_log_marg(lps, tgts, dummy):
    """
    lps - bsz x K
    tgts - bsz x max_crct
    dummy - bsz x 1, -inf
    Let's assume the dummy column is the zero'th column
    """
    # print(tgts.min().item(), tgts.max().item(), lps.size())
    plps = torch.cat([dummy, lps], 1) # bsz x 1+K
    crcts = plps.gather(1, tgts) # bsz x max_crct
    marglps = torch.logsumexp(crcts, 1) # can be -inf but will be ignored below
    return -marglps


def multi_binary_loss(scores, tgts, nopad_mask, avg=False):
    """
    scores - bsz x K
    tgts - bsz x max_crct; we assume dummy indices are 0 and everything else is +1
    nopad_mask - bsz x K; assumed to be 1 only for negative indices
    """
    pady = scores.new(scores.size(0), scores.size(1)+1).zero_() # bsz x K+1
    pady.scatter_(1, tgts, 1)
    y = pady[:, 1:] # ignores -1 => 0 tgt-padding...
    losses = F.binary_cross_entropy_with_logits(scores, y, reduction='none') # bsz x K
    negmask = 1 - y # 1 for negative examples or padding
    if avg:
        npos = y.sum(1)
        posloss = (y*losses).sum(1)/npos # bsz
    else:
        posloss = ((y*losses) + (negmask * 9999999)).min(1)[0] # bsz
    # get average negative example loss
    negmask.mul_(nopad_mask) # 1 for negative and not padding
    nneg = negmask.sum(1)
    negloss = (negmask*losses).sum(1)/nneg # bsz
    return posloss + negloss


def rec_loss(src, canv, encsrc, enccanv, startedmask, padidx, cosine=False):
    """
    src - srclen x bsz
    canv - canvlen x bsz
    encsrc - srclen x bsz x dim
    enccanv - canvlen x bsz x dim
    startedmask - bsz
    returns sum of averaged rec losses
    """
    finalmask = ~startedmask
    final_canvs = canv.t()[finalmask] # nfinal x canvlen
    final_enccanvs = enccanv.transpose(0, 1)[finalmask] # nfinal x canvlen x dim
    final_encsrcs = encsrc.transpose(0, 1)[finalmask] # nfinal x srclen x dim
    nfinal, srclen, dim = final_encsrcs.size()
    canvkeep = final_canvs != padidx # nfinal x canvlen
    cmask = canvkeep.float().div_(canvkeep.sum(1).view(-1, 1))
    maxpool = True
    if maxpool:
        ctxs = final_enccanvs.max(1)[0].unsqueeze(1).expand(nfinal, srclen, dim).contiguous()
    else:
        ctxs = cmask.unsqueeze(1).bmm(final_enccanvs).expand(nfinal, srclen, dim).contiguous()
    # only include non-padding in loss
    srckeep = src.t()[finalmask] != padidx # nfinal x srclen
    mmask = srckeep.float().div_(srckeep.sum(1).view(-1, 1)) # average over each src
    if cosine:
        losses = -F.cosine_similarity(final_encsrcs.view(-1, dim), ctxs.view(-1, dim))
    else:
        losses = F.mse_loss(
            final_encsrcs.view(-1, dim), ctxs.view(-1, dim), reduction='none').sum(1)
    return (losses*mmask.view(-1)).sum()


def discrec_loss(bwdmodel, src, canv, enccanv, startedmask, padidx):
    """
    src - srclen x bsz
    canv - canvlen x bsz
    enccanv - canvlen x bsz x dim
    startedmask - bsz
    """
    finalmask = ~startedmask
    final_canvs = canv.t()[finalmask].t() # canvlen x nfinal
    final_enccanvs = enccanv.transpose(0, 1)[finalmask].transpose(0, 1) # canvlen x nfinal x dim
    final_srcs = src.t()[finalmask].t().contiguous() # srclen x nfinal
    dec_states = bwdmodel(final_canvs, final_srcs, final_enccanvs, padidx) # srclen x nfinal x dim
    logits = dec_states.view(-1, dec_states.size(2)).mm(bwdmodel.lut.weight.t()) # srclen*nfinal x V
    srckeep = final_srcs != padidx # srclen x nfinal
    mmask = srckeep.float().div_(srckeep.sum(0).view(1, -1))
    loss = F.cross_entropy(logits, final_srcs.view(-1), reduction='none')
    return (loss*mmask.view(-1)).sum()


def zero_nan_hook(grad):
    indic = torch.any(torch.isnan(grad), dim=1)
    # zero out row with any nans in it...
    grad[indic.view(-1, 1).expand(-1, grad.size(1))] = 0
    return grad


class SEThing(nn.Module):
    def __init__(self, ngentypes, args):
        super().__init__()
        self.ngentypes = ngentypes
        self.C1lin, self.C2lin, self.N1lin, self.N2lin = None, None, None, None
        self.S1lin, self.S2lin, self.W1lin, self.W2lin = None, None, None, None

        if 'C' in args.Topts:
            self.C2lin = nn.Linear(args.embdim, args.embdim, bias=False)
            if 'x2' in args.Topts:
                self.C1lin = nn.Linear(args.embdim, args.embdim, bias=False)

        if 'N' in args.Topts:
            self.N2lin = nn.Linear(args.embdim, args.embdim, bias=False)
            if 'x2' in args.Topts:
                self.N1lin = nn.Linear(args.embdim, args.embdim, bias=False)

        if 'S' in args.Topts:
            self.S2lin = nn.Linear(args.embdim, args.embdim, bias=False)
            if 'x2' in args.Topts:
                self.S1lin = nn.Linear(args.embdim, args.embdim, bias=False)

        if 'W' in args.Topts:
            self.W2lin = nn.Linear(args.embdim, args.embdim, bias=False)
            if 'x2' in args.Topts:
                self.W1lin = nn.Linear(args.embdim, args.embdim, bias=False)

        # init things
        #self.init_lins(args)

    # def init_lins(self, args):
    #     lins = []
    #     if 'C' in args.Topts:
    #         lins.extend([self.C1lin, self.C2lin])
    #     if 'N' in args.Topts:
    #         lins.extend([self.N1lin, self.N2lin])
    #     if 'S' in args.Topts:
    #         lins.extend([self.S1lin, self.S2lin])
    #
    #     for lin in lins:
    #         lin.weight.data.uniform_(-args.init, args.init)
    #         if hasattr(lin, "bias") and lin.bias is not None:
    #             lin.bias.data.zero_()

    def get_start_lps(self, enccanv, encne, nemask, encsrc, src, outlut,
                      pad_idx=0, norm=False):
        """
        enccanv - canvlen x bsz x dim
        encne - max_nelen x nne x dim
        nemask - (1 or bsz) x max_nelen*nne bool tensor w/ 1s where we mask
        encsrc - srclen x bsz x dim
        src - srclen x bsz
        returns:
           bsz x canvlen*(nelen*nne+V+S) log probs
        """
        T, bsz, dim = enccanv.size()
        # get bsz x canvlen x dim start embs
        C = enccanv.transpose(0, 1).contiguous() if self.C1lin is None else self.C1lin(
            enccanv.transpose(0, 1))
        if norm and self.C1lin is not None:
            C = F.normalize(C, p=2, dim=2)

        nkeys = encne.view(-1, dim) if self.N1lin is None else self.N1lin(encne.view(-1, dim))
        if norm: # assuming always doing at least N2lin...
            nkeys = F.normalize(nkeys, p=2, dim=1)
        nscores = torch.mm(C.view(-1, dim), nkeys.t()) # bsz*canvlen x nelen*nne
        nscores = nscores.view(bsz, T, -1).masked_fill(
            nemask.unsqueeze(1).expand(bsz, T, nscores.size(1)), -float("inf"))

        skeys = encsrc if self.S1lin is None else self.S1lin(encsrc) # S x bsz x dim
        if norm:
            skeys = F.normalize(skeys, p=2, dim=2)
        sscores = C.bmm(skeys.permute(1, 2, 0)) # bsz x canvlen x S
        sscores = sscores.masked_fill((src.t() == pad_idx).unsqueeze(1).expand(
            bsz, T, src.size(0)), -float("inf"))

        wkeys = outlut.weight[:self.ngentypes] if self.W1lin is None else self.W1lin(
            outlut.weight[:self.ngentypes])
        if norm and self.W1lin is not None: # otherwise lut is already normalized
            wkeys = F.normalize(wkeys, p=2, dim=1)
        wscores = torch.mm(C.view(-1, dim), wkeys.t()) # bsz*canvlen x V

        # get bsz*canvlen x nelen*nne + V + S scores
        all_scores = torch.cat([nscores.view(bsz*T, -1), wscores, sscores.view(bsz*T, -1)], 1)
        start_lps = F.log_softmax(all_scores.view(bsz, -1), dim=1) # bsz x canvlen*(nelen*nne+V+S)
        return start_lps

    # def get_start_scores(self, enccanv, encne, nemask, encsrc, src, outlut, pad_idx=0):
    #     """
    #     enccanv - canvlen x bsz x dim
    #     encne - max_nelen x nne x dim
    #     nemask - (1 or bsz) x max_nelen*nne bool tensor w/ 1s where we mask
    #     encsrc - srclen x bsz x dim
    #     src - srclen x bsz
    #     returns:
    #        bsz x canvlen*(nelen*nne+V+S) logits and bsz x canvlen*(nelen*nne+V+S) nopad mask
    #     """
    #     T, bsz, dim = enccanv.size()
    #     # get bsz x canvlen x dim start embs
    #     C = enccanv.transpose(0, 1).contiguous() if self.C1lin is None else self.C1lin(
    #         enccanv.transpose(0, 1))
    #     nkeys = encne.view(-1, dim) if self.N1lin is None else self.N1lin(encne.view(-1, dim))
    #     nscores = torch.mm(C.view(-1, dim), nkeys.t()) # bsz*canvlen x nelen*nne
    #
    #     skeys = encsrc if self.S1lin is None else self.S1lin(encsrc) # S x bsz x dim
    #     sscores = C.bmm(skeys.permute(1, 2, 0)) # bsz x canvlen x S
    #
    #     wkeys = outlut.weight[:self.ngentypes] if self.W1lin is None else self.W1lin(
    #         outlut.weight[:self.ngentypes])
    #     wscores = torch.mm(C.view(-1, dim), wkeys.t()) # bsz*canvlen x V
    #
    #     # make bsz x canvlen x nelen*nne + V + S mask indicating non-padding
    #     nopad_mask = torch.cat([1 - nemask.unsqueeze(1).expand(bsz, T, nscores.size(1)).float(),
    #                             torch.ones_like(wscores).view(bsz, T, -1),
    #                             (src.t() != pad_idx).unsqueeze(1).expand(bsz, T, -1).float()], 2)
    #
    #     # get bsz*canvlen x nelen*nne + V + S scores
    #     all_scores = torch.cat([nscores.view(bsz*T, -1), wscores, sscores.view(bsz*T, -1)], 1)
    #     return all_scores.view(bsz, -1), nopad_mask.view(bsz, -1)

    def get_end_embs(self, encne, encsrc, outlut, netgts):
        """
        encne - max_nelen x nne x dim
        encsrc - srclen x bsz x dim
        returns max_len x bsz x dim embs
        """
        nkeys = encne if self.N2lin is None else self.N2lin(encne)
        skeys = encsrc if self.S2lin is None else self.S2lin(encsrc)
        wkeys = outlut.weight[:self.ngentypes] if self.W2lin is None else self.W2lin(
            outlut.weight[:self.ngentypes])
        all_keys = [nkeys, wkeys.unsqueeze(0), skeys]
        rem_embs = [all_keys[keytype][:, neidx] for (keytype, neidx, _, _, _, _, _) in netgts]
        padremaining_embs = pad_sequence(rem_embs) # max_len x bsz x dim
        return padremaining_embs

    # want p(neidx, l, r, j, k). idea: factorize as p(neidx, l, k)*p(r, tskip | neidx, l, j)
    # could just do same as above but zero out bad options, or could try to condition on
    # previous decision more directly.
    def get_end_lps1(self, enccanv, remembs, remmask, norm=False):
        """
        enccanv - canvlen x bsz x dim
        remembs - max_remlen x bsz x dim
        remmask - bsz x canvlen x remlen
        returns: bsz x canvlen*max_remlen
        """
        T, bsz, dim = enccanv.size()
        # get bsz x canvlen x dim end embs
        C = enccanv.transpose(0, 1) if self.C2lin is None else self.C2lin(enccanv.transpose(0, 1))
        if norm:
            remembs = F.normalize(remembs, p=2, dim=2)
            if self.C2lin is not None:
                C = F.normalize(C, p=2, dim=2)
        scores = C.bmm(remembs.permute(1, 2, 0)) # bsz x canvlen x maxremlen
        # mask out illegal end embs
        scores = scores.masked_fill(remmask, -float("inf"))
        return F.log_softmax(scores.view(bsz, -1), dim=1)

    # def get_end_scores1(self, enccanv, remembs):
    #     """
    #     enccanv - canvlen x bsz x dim
    #     remembs - max_remlen x bsz x dim
    #     returns: bsz x canvlen*max_remlen
    #     """
    #     T, bsz, dim = enccanv.size()
    #     # get bsz x canvlen x dim end embs
    #     C = enccanv.transpose(0, 1) if self.C2lin is None else self.C2lin(enccanv.transpose(0, 1))
    #     scores = C.bmm(remembs.permute(1, 2, 0)) # bsz x canvlen x maxremlen
    #     return scores.view(bsz, -1)


class LRSEThing(SEThing):
    def get_start_lps(self, lenccanv, encne, nemask, encsrc, src, outlut,
                      pad_idx=0, norm=False):
        """
        lenccanv - bsz x dim
        encne - max_nelen x nne x dim
        nemask - (1 or bsz) x max_nelen*nne bool tensor w/ 1s where we mask
        encsrc - srclen x bsz x dim
        src - srclen x bsz
        returns:
           bsz x (nelen*nne+V+S) log probs
        """
        bsz, dim = lenccanv.size()

        C = lenccanv if self.C1lin is None else self.C1lin(lenccanv) # bsz x dim start embs
        if norm and self.C1lin is not None:
            C = F.normalize(C, p=2, dim=1)

        nkeys = encne.view(-1, dim) if self.N1lin is None else self.N1lin(encne.view(-1, dim))
        if norm: # assuming always doing at least N2lin...
            nkeys = F.normalize(nkeys, p=2, dim=1)
        nscores = torch.mm(C, nkeys.t()) # bsz x nelen*nne
        nscores = nscores.masked_fill(nemask.expand(bsz, nscores.size(1)), -float("inf"))

        skeys = encsrc if self.S1lin is None else self.S1lin(encsrc) # S x bsz x dim
        if norm:
            skeys = F.normalize(skeys, p=2, dim=2)
        sscores = C.unsqueeze(1).bmm(skeys.permute(1, 2, 0)).squeeze(1) # bsz x 1 x S -> bsz x S
        sscores = sscores.masked_fill((src.t() == pad_idx), -float("inf"))

        wkeys = outlut.weight[:self.ngentypes] if self.W1lin is None else self.W1lin(
            outlut.weight[:self.ngentypes])
        if norm and self.W1lin is not None: # otherwise lut is already normalized
            wkeys = F.normalize(wkeys, p=2, dim=1)
        wscores = torch.mm(C, wkeys.t()) # bsz x V

        # get bsz*canvlen x nelen*nne + V + S scores
        all_scores = torch.cat([nscores, wscores, sscores], 1) # bsz x nelen*nne + V + S
        start_lps = F.log_softmax(all_scores, dim=1)
        return start_lps

    def get_end_lps1(self, lenccanv, remembs, remmask, norm=False):
        """
        lenccanv - bsz x dim
        remembs - max_remlen x bsz x dim
        remmask - bsz x remlen
        returns: bsz x max_remlen
        """
        bsz, dim = lenccanv.size()
        # get bsz x canvlen x dim end embs
        C = lenccanv if self.C2lin is None else self.C2lin(lenccanv)
        if norm:
            remembs = F.normalize(remembs, p=2, dim=2)
            if self.C2lin is not None:
                C = F.normalize(C, p=2, dim=1)
        scores = C.unsqueeze(1).bmm(  # bsz x 1 x maxremlen -> bsz x maxremlen
            remembs.permute(1, 2, 0)).squeeze(1)
        # mask out illegal end embs
        scores = scores.masked_fill(remmask, -float("inf"))
        return F.log_softmax(scores, dim=1)

    def get_end_lps2(self, lenccanv, encne, nemask, encsrc, src, outlut,
                     pad_idx=0, norm=False):
        """
        to be used during searhc....
        lenccanv - bsz x dim
        encne - max_nelen x nne x dim
        nemask - (1 or bsz) x max_nelen*nne bool tensor w/ 1s where we mask
        encsrc - srclen x bsz x dim
        src - srclen x bsz
        returns:
           bsz x (nelen*nne+V+S) log probs
        """
        bsz, dim = lenccanv.size()

        C = lenccanv if self.C2lin is None else self.C2lin(lenccanv) # bsz x dim start embs
        if norm and self.C2lin is not None:
            C = F.normalize(C, p=2, dim=1)

        nkeys = encne.view(-1, dim) if self.N2lin is None else self.N2lin(encne.view(-1, dim))
        if norm: # assuming always doing at least N2lin...
            nkeys = F.normalize(nkeys, p=2, dim=1)
        nscores = torch.mm(C, nkeys.t()) # bsz x nelen*nne
        nscores = nscores.masked_fill(nemask.expand(bsz, nscores.size(1)), -float("inf"))

        skeys = encsrc if self.S2lin is None else self.S2lin(encsrc) # S x bsz x dim
        if norm:
            skeys = F.normalize(skeys, p=2, dim=2)
        sscores = C.unsqueeze(1).bmm(skeys.permute(1, 2, 0)).squeeze(1) # bsz x 1 x S -> bsz x S
        sscores = sscores.masked_fill((src.t() == pad_idx), -float("inf"))

        wkeys = outlut.weight[:self.ngentypes] if self.W2lin is None else self.W2lin(
            outlut.weight[:self.ngentypes])
        if norm and self.W2lin is not None: # otherwise lut is already normalized
            wkeys = F.normalize(wkeys, p=2, dim=1)
        wscores = torch.mm(C, wkeys.t()) # bsz x V

        # get bsz*canvlen x nelen*nne + V + S scores
        all_scores = torch.cat([nscores, wscores, sscores], 1) # bsz x nelen*nne + V + S
        end_lps = F.log_softmax(all_scores, dim=1)
        return end_lps


class RelMvIdxEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings+1, embedding_dim, padding_idx=num_embeddings)
        self.rul_pad_idx = -1

    def forward(self, inputs):
        """
        inputs - bsz x seqlen (N.B. bsz first!)
        ASSUMES padding w/ -1
        """
        mask = inputs == self.rul_pad_idx
        maxes = inputs.max(1)[0] # bsz
        relinputs = maxes.view(-1, 1) - inputs
        return super(RelMvIdxEmbedding, self).forward(
            relinputs.masked_fill(mask, self.padding_idx))


# below based on https://huggingface.co/transformers/_modules/transformers/modeling_bart.html#BartModel
def set_up_config(bosidx, eosidx, padidx, ntypes, args):
    config = transformers.BartConfig()
    config.bos_token_id = bosidx
    config.d_model = args.embdim
    config.decoder_attention_heads = args.nheads
    config.decoder_ffn_dim = args.ffdim
    config.decoder_layers = args.enc_layers
    config.dropout = args.drop
    config.encoder_attention_heads = args.nheads
    config.encoder_ffn_dim = args.ffdim
    config.encoder_layers = args.senc_layers
    config.eos_token_id = eosidx
    config.normalize_before = args.prenorm
    config.num_hidden_layers = args.enc_layers # I think not used
    config.pad_token_id = padidx
    config.extra_pos_embeddings = padidx + 1 # this way everything gets shifted past pad
    config.vocab_size = ntypes
    return config


class BartThing(nn.Module):
    def __init__(self, ntypes, ngentypes, args):
        super().__init__()
        # max_len = 500
        self.ngentypes = ngentypes
        maxnorm = 1.0 if args.norm else None
        self.lut = nn.Embedding(ntypes, args.embdim, padding_idx=args.padidx,
                                max_norm=maxnorm)
        self.src_mode = args.src_mode
        self.flut = nn.Embedding(4, args.embdim, padding_idx=3)
        self.rlut = RelMvIdxEmbedding(100, args.embdim) # assuming no more than 100 moves
        config = set_up_config(args.bosidx, args.eosidx, args.padidx, ntypes, args)
        self.config = config
        self.encoder = BartEncoder(config, self.lut)
        self.decoder = BartDecoder(config, self.lut)
        if args.share_encs:
            self.ne_encoder = self.encoder
        else:
            self.ne_encoder = BartEncoder(config, self.lut)
        if args.leftright:
            self.actmodel = LRSEThing(ngentypes, args)
        else:
            self.actmodel = SEThing(ngentypes, args)
        if args.recloss == "disc":
            self.bwdmodel = BartRecEncDec(ntypes, self.lut, args)
        self.init_weights()

    # copied from transformers/BART
    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def init_weights(self):
        self.apply(self._init_weights)

    # since we wanna use more features, we'll bypass the official enc forward method
    def enc_fwd(self, encoder, src, usedfeats, pad_idx, neenc=False):
        """
        src - srclen x bsz
        usedfeats - srclen x bsz
        """
        srcemb = self.lut(src.t()) + encoder.embed_positions(src.t()) # bsz x srclen x dim
        if neenc:
            if self.src_mode == "mask":
                srcemb = srcemb + self.flut.weight[2].view(1, 1, -1) # 0 and 1 for src
            elif self.src_mode == "feat":
                srcemb = srcemb + self.flut.weight[1].view(1, 1, -1)
        else:
            if self.src_mode == "mask":
                srcemb = srcemb + self.flut(usedfeats.t())
            elif self.src_mode == "feat":
                srcemb = srcemb + self.flut.weight[0].view(1, 1, -1)
        srcemb = F.dropout(encoder.layernorm_embedding(srcemb),
                           p=encoder.dropout, training=encoder.training)
        x = srcemb.transpose(0, 1) # srclen x bsz x dim
        attention_mask = src.t() == pad_idx

        for encoder_layer in encoder.layers:
            # dropout_probability = random.uniform(0, 1)
            # if encoder.training and (dropout_probability < encoder.layerdrop):
            #     pass
            # else:
            x, attn = encoder_layer(x, attention_mask, output_attentions=False)

        if encoder.layer_norm:
            x = encoder.layer_norm(x)
        encoder_hidden_states, encoder_padding_mask = x, attention_mask
        return encoder_hidden_states, encoder_padding_mask

    # again bypassing the official dec forward method
    def dec_fwd(self, lengths, canv, relidxs, pad_idx,
                encoder_hidden_states, encoder_padding_mask):
        """
        canv - canvlen x bsz
        lengths - canvlen x bsz
        relidxs - canvlen x bsz
        """
        canvemb = self.lut(canv.t()) + self.decoder.embed_positions(canv.t()) # bsz x canvlen x dim
        canvemb = canvemb + self.rlut(relidxs.t())
        # if hasattr(self, "llut"):
        #     canvemb = canvemb + self.llut(lengths)
        canvemb = F.dropout(self.decoder.layernorm_embedding(canvemb),
                            p=self.decoder.dropout, training=self.decoder.training)
        x = canvemb.transpose(0, 1) # canvlen x bsz x dim
        decoder_padding_mask = canv.t() == pad_idx
        decoder_causal_mask = None

        for idx, decoder_layer in enumerate(self.decoder.layers):
            # dropout_probability = random.uniform(0, 1)
            # if self.decoder.training and (dropout_probability < self.decoder.layerdrop):
            #     continue
            #layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=None,
                #layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=False,
            )
            # if use_cache:
            #     next_decoder_cache.append(layer_past.copy())
            if self.decoder.layer_norm and (idx == len(self.decoder.layers) - 1): # last layer
                x = self.decoder.layer_norm(x)
            # if output_attentions:
            #     all_self_attns += (layer_self_attn,)
        return x

    def src_encode(self, src, usedfeats, lengths, canv, relidxs, pad_idx,
                   memory=None, srckeymask=None):
        """
        src - srclen x bsz
        usedfeats - srclen x bsz
        canv - canvlen x bsz
        lengths - canvlen x bsz
        returns:
            srclen x bsz x dim, canvlen x bsz x dim
        """
        if memory is None:
            encoder_hidden_states, encoder_padding_mask = self.enc_fwd(
                self.encoder, src, usedfeats, pad_idx, neenc=False)
        else:
            encoder_hidden_states, encoder_padding_mask = memory, srckeymask

        dec_hidden_states = self.dec_fwd(
            lengths, canv, relidxs, pad_idx, encoder_hidden_states, encoder_padding_mask)

        return encoder_hidden_states, dec_hidden_states, encoder_padding_mask

    def ne_encode(self, neighbs, pad_idx):
        """
        neighbs - max_nelen x nne
        """
        encoder_hidden_states, _ = self.enc_fwd(
            self.ne_encoder, neighbs, None, pad_idx, neenc=True)

        return encoder_hidden_states


class TokenCopyBart(BartThing):
    def __init__(self, ntypes, ngentypes, args):
        args.src_mode = "feat"
        super().__init__(ntypes, ngentypes, args)
        del self.rlut
        del self.actmodel
        self.Clin = nn.Linear(args.embdim, args.embdim, bias=False)
        self.Nlin = nn.Linear(args.embdim, args.embdim, bias=False)
        self.Slin = nn.Linear(args.embdim, args.embdim, bias=False)
        self.Wlin = nn.Linear(args.embdim, args.embdim, bias=False)
        for mod in [self.Clin, self.Nlin, self.Slin, self.Wlin]:
            self._init_weights(mod)

    def dec_fwd(self, tgtinp, pad_idx, encoder_hidden_states, encoder_padding_mask):
        """
        same as parent but uses causal mask and no relidxs
        """
        tgtlen = tgtinp.size(0)
        tgtemb = self.lut(tgtinp.t()) + self.decoder.embed_positions(tgtinp.t()) # bsz x tlen x dim
        tgtemb = F.dropout(self.decoder.layernorm_embedding(tgtemb),
                           p=self.decoder.dropout, training=self.decoder.training)
        x = tgtemb.transpose(0, 1) # tgtlen x bsz x dim
        decoder_padding_mask = tgtinp.t() == pad_idx
        # copied from bart code
        decoder_causal_mask = x.new(tgtlen, tgtlen).fill_(float("-inf"))
        maskidxs = torch.arange(tgtlen).to(x.device)
        decoder_causal_mask.masked_fill_(maskidxs < (maskidxs + 1).view(tgtlen, 1), 0)

        for idx, decoder_layer in enumerate(self.decoder.layers):
            # dropout_probability = random.uniform(0, 1)
            # if self.decoder.training and (dropout_probability < self.decoder.layerdrop):
            #     continue
            #layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=None,
                #layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=False,
            )
            # if use_cache:
            #     next_decoder_cache.append(layer_past.copy())
            if self.decoder.layer_norm and (idx == len(self.decoder.layers) - 1): # last layer
                x = self.decoder.layer_norm(x)
            # if output_attentions:
            #     all_self_attns += (layer_self_attn,)
        return x

    def forward(self, srcs, tgtinps, nes, pad_idx):
        """
        returns tgtlen*bsz x nelen*nne+V+S log probs
        """
        encsrc, srcmask = self.enc_fwd( # srclen x bsz x dim, bsz x srclen
            self.encoder, srcs, None, pad_idx, neenc=False)
        enctgt = self.dec_fwd(tgtinps, pad_idx, encsrc, srcmask) # tgtlen x bsz x dim
        encne = self.ne_encode(nes, pad_idx) # nelen x nne x dim
        # this is similar to SE architecture
        T, bsz, dim = enctgt.size()
        C = self.Clin(enctgt.view(T*bsz, dim)) # tgtlen*bsz x dim
        nkeys = self.Nlin(encne.view(-1, dim)) # nelen*nne x dim
        nscores = C.mm(nkeys.t()) # tgtlen*bsz x nelen*nne
        nemask = (nes.view(-1) == pad_idx).unsqueeze(0) # 1 x nelen*nne
        nscores = nscores.masked_fill(nemask.expand(T*bsz, nscores.size(1)), -float("inf"))

        skeys = self.Slin(encsrc) # srclen x bsz x dim
        sscores = C.view(T, bsz, dim).transpose(0, 1).bmm( # bsz x tgtlen x srclen
            skeys.permute(1, 2, 0)).transpose(0, 1)        # -> tgtlen x bsz x srclen
        sscores = sscores.masked_fill((srcs.t() == pad_idx).unsqueeze(0).expand(
            T, bsz, srcs.size(0)), -float("inf"))

        wkeys = self.Wlin(self.lut.weight[:self.ngentypes]) # ngentypes x dim
        wscores = C.mm(wkeys.t()) # tgtlen*bsz x V
        all_scores = torch.cat([nscores, wscores, sscores.view(T*bsz, -1)], 1)
        lps = F.log_softmax(all_scores, dim=1)
        return lps


    # standard decoder; no transformations except on src, shares embs
    def none_fwd(self, srcs, tgtinps, pad_idx):
        """
        returns tgtlen*bsz x V+S log probs
        """
        encsrc, srcmask = self.enc_fwd( # srclen x bsz x dim, bsz x srclen
            self.encoder, srcs, None, pad_idx, neenc=False)
        enctgt = self.dec_fwd(tgtinps, pad_idx, encsrc, srcmask) # tgtlen x bsz x dim
        C = enctgt
        T, bsz, dim = enctgt.size()

        skeys = self.Slin(encsrc) # srclen x bsz x dim
        sscores = C.transpose(0, 1).bmm( # bsz x tgtlen x srclen
            skeys.permute(1, 2, 0)).transpose(0, 1)  # -> tgtlen x bsz x srclen
        sscores = sscores.masked_fill((srcs.t() == pad_idx).unsqueeze(0).expand(
            T, bsz, srcs.size(0)), -float("inf"))

        # wkeys = self.Wlin(self.lut.weight[:self.ngentypes]) # ngentypes x dim
        wkeys = self.lut.weight[:self.ngentypes]
        wscores = C.view(-1, C.size(2)).mm(wkeys.t()) # tgtlen*bsz x V
        all_scores = torch.cat([wscores, sscores.view(T*bsz, -1)], 1)
        lps = F.log_softmax(all_scores, dim=1)
        return lps
