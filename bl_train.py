import os
import gc
import math
import time
import argparse

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW as HFAdamW, get_linear_schedule_with_warmup

import db2 as data
import model_utils as mutils

def do_batch(model, db, masks, dummy, device, args, val=False):
    try:
        min_nes = args.min_valnes if val else args.min_nes
        srcs, neighbs, tgtinps, tgtidxs = db.do_bl_batch(min_nes, val=val)
        _, bsz = srcs.size()

        srcs, neighbs = srcs.to(device), neighbs.to(device)
        tgtinps, tgtidxs = tgtinps.to(device), tgtidxs.to(device)

        if min_nes > 0:
            lps = model(srcs, tgtinps, neighbs, db.pad_idx)
        else:
            lps = model.none_fwd(srcs, tgtinps, db.pad_idx)

        loss = mutils.neg_log_marg(lps, tgtidxs+1, dummy.expand(lps.size(0), 1))
        tgtmask = tgtinps.view(-1) != db.pad_idx
        loss = loss[tgtmask].sum()

        if not val: # backprop but don't divide here..
            loss.backward()

    except RuntimeError as ex:
        # raise ex
        print("assuming OOM")
        gc.collect()
        torch.cuda.empty_cache()
        loss = None

    return loss, bsz


# this does gold roll-in training
def train(db, model, optim, scheduler, masks, device, args):
    model.train()
    total_loss = 0
    nex = 0
    dummy = torch.Tensor([[-float("inf")]]).to(device)
    optim.zero_grad()
    accum_size = 0
    for i in range(args.mbs_per_epoch):
        loss, bsz = do_batch(model, db, masks, dummy, device, args)
        if loss is None: # memory issue
            continue
        if torch.isnan(loss):
            print("got loss nan on", i, "...bailing")
            break
        total_loss += loss.item()
        accum_size += bsz

        if accum_size >= args.min_seq_accum or i == args.mbs_per_epoch-1:
            for p in model.parameters(): # avg grads
                if p.grad is not None:
                    p.grad.data.div_(accum_size)
            accum_size = 0
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optim.step()
            optim.zero_grad()
            scheduler.step()

        nex += bsz
        if (i+1) % args.log_interval == 0:
            print("{:5d}/{:5d} | lr {:02.4f} | loss {:7.2f}".format(
                i+1, args.mbs_per_epoch, scheduler.get_last_lr()[0], total_loss/nex))

    return total_loss/nex


def validate(db, model, masks, device, args):
    model.eval()
    total_loss = 0
    nex = 0
    dummy = torch.Tensor([[-float("inf")]]).to(device)
    db.val_bidx = 0
    for i in range(args.val_mbs_per_epoch):
        loss, bsz = do_batch(model, db, masks, dummy, device, args, val=True)
        if loss is None: # memory issue
            continue
        total_loss += loss.item()
        nex += bsz
    return total_loss, nex


parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', type=str, default="data/wb", help='datadir')
parser.add_argument('-vocopts', nargs='+', type=int, default=[20, 20, None, None],
                    help='missing_thresh,reg_thresh,max_gen_voc_size,max_voc_size')
parser.add_argument('-flat_moves', action='store_true', help='')
parser.add_argument('-enclose', action='store_true', help='')
parser.add_argument('-sel_firstlast_idxing', action='store_true', help='')
parser.add_argument('-leftright', action='store_true', help='not used')
parser.add_argument('-nne', type=int, default=100,
                    help='neighbors per example')
parser.add_argument("-prote_fi", default="", type=str, help="")
parser.add_argument("-tokfi",
                    default=None, type=str, help="")
parser.add_argument("-split_dashes", action='store_true', help="")
parser.add_argument('-min_nes', type=int, default=100, help='per example')
parser.add_argument('-min_valnes', type=int, default=20, help='per example')
parser.add_argument('-use_pttf', action='store_true', help='')
parser.add_argument('-decomp_mode', type=str, default='SE', choices=['SE', 'IS'], help='')
parser.add_argument('-prenorm', action='store_true', help='')
parser.add_argument('-embdim', type=int, default=512, help='')
parser.add_argument('-ffdim', type=int, default=1024, help='tranformer internal dim')
parser.add_argument('-nheads', type=int, default=8, help='')
parser.add_argument('-senc_layers', type=int, default=4, help='')
parser.add_argument('-enc_layers', type=int, default=6, help='')
parser.add_argument('-norm', action='store_true', help='normalize embeddings')
parser.add_argument('-fixed_pos_embs', action='store_true', help='')
parser.add_argument('-max_moves', type=int, default=100, help='')
parser.add_argument('-max_canvlen', type=int, default=200, help='helps w/ mem...')
parser.add_argument('-use_lengths', action='store_true', help='')
parser.add_argument('-share_encs', action='store_true', help='')
parser.add_argument('-activ', type=str, default='gelu', choices=['gelu', 'relu'], help='')
parser.add_argument('-Topts', type=str, default='NSW',
                    choices=['NSW', 'NSWx2', 'CNSW', 'CNSWx2'], help='')
parser.add_argument('-optalg', type=str, default='adamw', choices=['hf_adamw', 'adamw'], help='')
parser.add_argument('-init', type=float, default=0.1, help='param init')
parser.add_argument('-adamhyps', type=str, default='0.9,0.999,1e-8,0.001', help='')
parser.add_argument('-lr', type=float, default=0.0005, help='initial learning rate')
parser.add_argument('-no_isr_schedule', action='store_true', help='')
parser.add_argument('-no_decay', action='store_true', help='')
parser.add_argument('-warmup_init_lr', type=float, default=1e-7, help='initial learning rate')
parser.add_argument('-warmup_steps', type=int, default=4000, help='')
parser.add_argument('-clip', type=float, default=1, help='gradient clipping')
parser.add_argument('-epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('-bsz', type=int, default=32, help='batch size')
parser.add_argument('-val_bsz', type=int, default=32, help='batch size')
parser.add_argument('-min_seq_accum', type=int, default=200, help='')
parser.add_argument('-drop', type=float, default=0.1, help='dropout')
parser.add_argument('-mbs_per_epoch', type=int, default=500000000, help='')
parser.add_argument('-val_mbs_per_epoch', type=int, default=500000000, help='')
#parser.add_argument('-losswts', nargs='+', type=float, default=[0.5, 0.5, 0.0], help='')
parser.add_argument('-recloss', type=str, default=None, choices=['cosine', 'l2', 'disc'], help='')
parser.add_argument('-seed', type=int, default=3636, help='random seed')
parser.add_argument('-wait', type=int, default=3, help='')
parser.add_argument('-cuda', action='store_true', help='use CUDA')
parser.add_argument('-log_interval', type=int, default=200, help='report interval')
parser.add_argument('-save', type=str, default='', help='path to save the final model')
parser.add_argument('-train_from', type=str, default='', help='')
parser.add_argument('-just_eval', action='store_true', help='')

# adapted from huggingface transformers examples/lightning_base.py
def prep_optim(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.awd,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,},]
    if args.optalg == "hf_adamw":
        optim = HFAdamW(grouped_parameters, lr=args.lr, betas=(args.beta1, args.beta2),
                        eps=args.aeps)
    else:
        optim = torch.optim.AdamW(grouped_parameters, lr=args.lr, betas=(args.beta1, args.beta2),
                                  eps=args.aeps)

    if args.no_isr_schedule:
        lr_lambda = lambda current_step: 1
    else:
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                lr_step = (args.lr - args.warmup_init_lr)/args.warmup_steps
                return (args.warmup_init_lr + current_step*lr_step)/args.lr
            return args.warmup_steps**0.5 * current_step**-0.5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    return optim, scheduler

def main(db, args):
    print("main args", args)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with -cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    print("total train batches", db.nbatches)
    print("total val batches", db.nval_batches)

    args.padidx = db.d.w2i["<pad>"]
    args.bosidx = db.d.w2i["<bos>"]
    args.eosidx = db.d.w2i["<eos>"]
    mod_ctor = mutils.TokenCopyBart

    if args.train_from:
        saved_stuff = torch.load(args.train_from)
        saved_args = saved_stuff["opt"]
        model = mod_ctor(len(db.d), db.d.gen_voc_size, saved_args)
        bestmodel = mod_ctor(len(db.d), db.d.gen_voc_size, saved_args)
        model.load_state_dict(saved_stuff["sd"])
        model = model.to(device)
        optim, scheduler = prep_optim(model, saved_args)
        optim.load_state_dict(saved_stuff["osd"])
        scheduler.load_state_dict(saved_stuff["ssd"])
        best_loss, best_acc = saved_stuff["bestloss"], saved_stuff["bestacc"]
        # update things that could reasonably change when restarting...
        saved_args.epochs, saved_args.mbs_per_epoch = args.epochs, args.mbs_per_epoch
        saved_args.val_mbs_per_epoch, saved_args.save = args.val_mbs_per_epoch, args.save
        saved_args.bsz, saved_args.wait = args.bsz, args.wait
        saved_args.just_eval = args.just_eval
        args = saved_args
        print("starting with:", scheduler._step_count, saved_args.lr, scheduler.get_last_lr(),
              best_loss, best_acc)
        #assert False
    else:
        model = mod_ctor(len(db.d), db.d.gen_voc_size, args).to(device)
        bestmodel = mod_ctor(len(db.d), db.d.gen_voc_size, args)
        optim, scheduler = prep_optim(model, args)
        best_loss, best_acc = float("inf"), 0

    # max_ncanvs, max_seqlen = 250, max(db.max_srclen, db.max_tgtlen)
    # emask = torch.ones(max_ncanvs, args.max_canvlen, max_seqlen, dtype=torch.bool).to(device)
    # masks = [emask]
    masks = None

    # if args.just_eval:
    #     db.curr_batch = None
    #     with torch.no_grad():
    #         vloss1, vloss2, vnex, avg_acc = validate(db, model, masks, device, args)
    #         voloss = (vloss1 + vloss2)/vnex
    #         print("Epoch {:3d} | val loss1 {:6.3f} | val loss2 {:6.3f} | "
    #               "val loss {:6.3f} | avg acc {:6.3f}".format(
    #                   0, vloss1/vnex, vloss2/vnex, voloss, avg_acc))
    #     return None, 0, None, None

    #assert args.losswts[2] > 0 or args.recloss is None
    bad_epochs = -1
    for ep in range(args.epochs):
        trloss = train(db, model, optim, scheduler, masks, device, args)
        if trloss is None:
            print("we're done here")
            break
        print("Epoch {:3d} | train loss {:6.3f}".format(ep, trloss))

        with torch.no_grad():
            vloss, vnex  = validate(db, model, masks, device, args)
            voloss = vloss/vnex
            print("Epoch {:3d} | val loss {:6.3f}".format(ep, voloss))

        if voloss < best_loss:
            best_loss = voloss
            # if avg_acc > best_acc:
            #     best_acc = avg_acc
            #     if os.path.exists(args.save+"-a"): # we should delete it since we've surpassed it
            #         os.remove(args.save+"-a")
            bad_epochs = -1
            print("updating best model")
            bestmodel.load_state_dict(model.state_dict())
            if len(args.save) > 0:
                savepath = args.save+"-l"
                print("saving model to", savepath)
                torch.save(
                    {"opt": args, "sd": bestmodel.state_dict(), "osd": optim.state_dict(),
                     "ssd": scheduler.state_dict(), "bestloss": best_loss, "bestacc": -1},
                    savepath)

        bad_epochs += 1
        if bad_epochs >= args.wait:
            break
        print("")
    return bestmodel, best_loss, optim, scheduler


if __name__ == "__main__":
    args = parser.parse_args()
    args.leftright = False
    args.sel_firstlast_idxing = True
    args.arbl = True
    print(args)

    db = data.TrainDB(args)
    #assert False

    beta1, beta2, aeps, awd = [float(thing) for thing in args.adamhyps.split(',')]
    args.beta1, args.beta2, args.aeps, args.awd = beta1, beta2, aeps, awd
    torch.manual_seed(args.seed)
    bestmodel, runloss, optim, scheduler = main(db, args)
