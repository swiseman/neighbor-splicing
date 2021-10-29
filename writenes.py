import argparse

from collections import defaultdict, Counter

import torch

from utils import get_wikibio_fields, get_e2e_fieldspp

def make_dicts(ne_fi, get_fields):
    keycount = Counter()
    valcount = Counter()
    with open(ne_fi) as f:
        for line in f:
            fields = get_fields(line.strip().split())
            keycount.update(fields.keys())
            valcount.update([tuple(v) for v in fields.values()])
    for key in list(keycount.keys()):
        if keycount[key] < 20:
            del keycount[key]
    for key in list(valcount.keys()):
        if valcount[key] < 200:
            del valcount[key]
    keyi2s = list(keycount.keys())
    keys2i = {s: i for i, s in enumerate(keyi2s)}
    vali2s = list(valcount.keys())
    vals2i = {s: i for i, s in enumerate(vali2s)}
    return keyi2s, keys2i, vali2s, vals2i

def make_spmats(ne_fi, keys2i, vals2i, get_fields):
    fieldrows, fieldcols = [], []
    valrows, valcols = [], []
    fieldrowsums = []
    nrows = 0
    with open(ne_fi) as f:
        for i, line in enumerate(f):
            fields = get_fields(line.strip().split())
            valz = set()
            [valz.update(v) for v in fields.values()]
            fcols = [keys2i[thing] for thing in fields.keys() if thing in keys2i]
            frows = [i]*len(fcols)
            fieldrowsums.append(len(fcols))
            fieldrows.extend(frows)
            fieldcols.extend(fcols)
            vcols = [vals2i[thing] for thing in valz if thing in vals2i]
            vrows = [i]*len(vcols)
            valrows.extend(vrows)
            valcols.extend(vcols)
            nrows += 1
    fieldmat = torch.sparse.FloatTensor(torch.LongTensor([fieldrows, fieldcols]),
                                        torch.ones(len(fieldrows)),
                                        torch.Size([nrows, len(keys2i)]))
    valmat = torch.sparse.FloatTensor(torch.LongTensor([valrows, valcols]),
                                      torch.ones(len(valrows)),
                                      torch.Size([nrows, len(vals2i)]))
    return fieldmat, valmat, torch.Tensor(fieldrowsums)

def make_dense_mats(ne_fi, keys2i, get_fields, vals2i=None, dense=True):
    mat, vmat = [], []
    with open(ne_fi) as f:
        for i, line in enumerate(f):
            fields = get_fields(line.strip().split())
            fcols = [keys2i[thing] for thing in fields.keys() if thing in keys2i]
            if dense:
                row = torch.zeros(len(keys2i))
                row[torch.LongTensor(fcols)] = 1
                mat.append(row)
            else:
                mat.append(fcols)
            if vals2i is not None:
                vcols = [vals2i[tuple(thing)] for thing in fields.values()
                         if tuple(thing) in vals2i]
                if dense:
                    vrow = torch.zeros(len(vals2i))
                    vrow[torch.LongTensor(vcols)] = 1
                    vmat.append(vrow)
                else:
                    vmat.append(vcols)
    if dense:
        if vals2i is not None:
            mat, vmat = torch.stack(mat), torch.stack(vmat)
        else:
            mat, vmat = torch.stack(mat), None
    return mat, vmat

def get_f(O, B, rowsums):
    prec = O/rowsums.view(1, -1) # bsz x nex
    rec = O/B.sum(1).add_(1e-6).view(-1, 1) # bsz x nex
    F = 2*prec
    F.mul_(rec)
    prec.add_(rec)
    prec.add_(1e-6)
    F.div_(prec)
    return F

parser = argparse.ArgumentParser()
parser.add_argument("-ne_fi", default=None, type=str, help="should be src side")
parser.add_argument("-train_tgt_fi", default=None, type=str,
                    help="tgts of neighbors (typically from train)")
parser.add_argument("-val_src_fi", default=None, type=str,
                    help="src side of what we're translating")
parser.add_argument("-out_fi", default=None, type=str, help="")
parser.add_argument('-nne', type=int, default=500, help='')
parser.add_argument('-bsz', type=int, default=1024, help='')
parser.add_argument("-wrkr", default="1,1", type=str, help="")
parser.add_argument('-cuda', action='store_true', help='use CUDA')
parser.add_argument('-e2e', action='store_true', help='')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.e2e:
        get_fields = get_e2e_fieldspp
    else:
        get_fields = get_wikibio_fields

    # get unigrams
    tgtcounter = Counter()

    # get unigram freqs
    with open(args.train_tgt_fi) as f:
        for line in f:
            tgtcounter.update(line.strip().split())

    # get avg unigram freq
    aufs = []
    with open(args.train_tgt_fi) as f:
        for line in f:
            tokes = line.strip().split()
            avg_ufreq = sum(tgtcounter[toke] for toke in tokes)/len(tokes)
            aufs.append(avg_ufreq)
    aufs = torch.Tensor(aufs).to(device)
    # normalize and then multiply by 0.001
    aufs.div_(aufs.max(0)[0]*100)

    keyi2s, keys2i, vali2s, vals2i = make_dicts(args.ne_fi, get_fields)
    print("made dicts")
    print("btw", len(keyi2s), len(vali2s))

    fieldmat, valmat = make_dense_mats(args.ne_fi, keys2i, get_fields, vals2i, dense=True)
    fieldmat, valmat = fieldmat.to(device), valmat.to(device)
    rowsums, vrowsums = fieldmat.sum(1), valmat.sum(1) # nex
    rowsums.add_(1e-6)
    vrowsums.add_(1e-6)
    assert aufs.size(0) == rowsums.size(0)
    assert rowsums.size(0) == vrowsums.size(0)

    val_src_fi = args.val_src_fi if args.val_src_fi is not None else args.ne_fi

    valkeyidxs, valvalidxs = make_dense_mats(val_src_fi, keys2i, get_fields, vals2i, dense=False)
    print("got stuff again")

    with torch.no_grad():
        Bk = torch.zeros(args.bsz, len(keys2i)).to(device)
        Bv = torch.zeros(args.bsz, len(vals2i)).to(device)
        with open(args.out_fi, "w+") as f:
            for i in range(0, len(valkeyidxs), args.bsz):
                # make a batch
                B = Bk[:min(args.bsz, len(valkeyidxs)-i)]
                B2 = Bv[:min(args.bsz, len(valkeyidxs)-i)]
                B.zero_()
                B2.zero_()
                for j in range(i, min(i+args.bsz, len(valkeyidxs))):
                    B[j-i][torch.LongTensor(valkeyidxs[j])] = 1
                    B2[j-i][torch.LongTensor(valvalidxs[j])] = 1
                O = fieldmat.mm(B.t()).t() # bsz x nex
                fscores = get_f(O, B, rowsums)
                O2 = valmat.mm(B2.t()).t()
                fscores2 = get_f(O2, B2, vrowsums)
                # want to break ties by values, and then by unigrams
                fscores.mul_(100)
                fscores.add_(fscores2)
                fscores.add_(aufs.view(1, -1))
                tops, argtops = torch.topk(fscores, args.nne, dim=1)
                for j in range(fscores.size(0)):
                    f.write(" ".join(["%d,%.2f" % (argtops[j][k].item(), tops[j][k].item())
                                      for k in range(args.nne)]))
                    f.write("\n")
