import os
from collections import defaultdict, OrderedDict

import regex

import torch

########## Data stuff ################
def dashrep(matchobj):
    l, r = matchobj.span(0)
    nustring = "%s - %s" % (matchobj.string[l], matchobj.string[r-1])
    return nustring

def get_neidxs_text(datadir, nefi, nne, val=False):
    neidxs = []
    with open(os.path.join(datadir, nefi)) as f:
        for l, line in enumerate(f):
            if val: # using training neighbors for validation
                netups = line.strip().split()[:nne]
                idxs = [int(thing.split(',')[0]) for thing in netups]
            else:
                netups = line.strip().split()[:nne+1] # not sure if self is inside
                idxs = [int(thing.split(',')[0]) for thing in netups]
                idxs = [thing for thing in idxs if thing != l][:nne]
            neidxs.append(torch.IntTensor(idxs))
    return torch.stack(neidxs)

def get_neidxs_dist_text(datadir, nefi, nne, val=False):
    neidxs, nedists = [], []
    with open(os.path.join(datadir, nefi)) as f:
        for l, line in enumerate(f):
            idxs, dists = [], []
            if val: # using training neighbors for validation
                netups = line.strip().split()[:nne]
                for thing in netups:
                    pieces = thing.split(',')
                    idxs.append(int(pieces[0]))
                    dists.append(float(pieces[1]))
            else:
                netups = line.strip().split()[:nne+1] # not sure if self is inside
                for thing in netups:
                    pieces = thing.split(',')
                    idx = int(pieces[0])
                    if idx != l and len(idxs) < nne:
                        idxs.append(idx)
                        dists.append(float(pieces[1]))
            neidxs.append(idxs)
            nedists.append(dists)
    return neidxs, nedists

def get_neidxs_pt(datadir, nefi, nne, val=False):
    if val:
        neidxs = torch.load(os.path.join(datadir, nefi))[:, :nne]
    else:
        neidxs = torch.load(os.path.join(datadir, nefi))[:, :nne+1]
        for i in range(neidxs.size(0)):
            nurow = neidxs[i][neidxs[i] != i]
            if nurow.size(0) == nne: # i must be in first nne+1
                neidxs[i][:nne] = nurow
        neidxs = neidxs[:, :nne]
    return neidxs

def get_dists_pt(datadir, nefi, nne):
    """
    assumes val=True
    """
    distfi = nefi.split('.')[0] + "-dists.pt"
    dists = torch.load(os.path.join(datadir, distfi))[:, :nne]
    return dists

def get_neidxs(datadir, nefi, nne, val=False, dist=False):
    try:
        with open(os.path.join(datadir, nefi)) as f:
            for line in f:
                istext = True
                break
    except UnicodeDecodeError:
        istext = False
    if istext:
        if dist:
            return get_neidxs_dist_text(datadir, nefi, nne, val=val)
        return get_neidxs_text(datadir, nefi, nne, val=val)
    if dist:
        neidxs = get_neidxs_pt(datadir, nefi, nne, val=val)
        dists = get_dists_pt(datadir, nefi, nne)
        assert dists.size(1) == neidxs.size(1)
        if dists.size() != neidxs.size():
            assert dists.size(0) < neidxs.size(0)
            neidxs = neidxs[:dists.size(0)]
        return neidxs.tolist(), dists.tolist()
    return get_neidxs_pt(datadir, nefi, nne, val=val)

def get_wikibio_fieldsold(tokes):
    """
    key -> list of words
    """
    fields = OrderedDict()
    key = None # remembers last key
    for toke in tokes:
        if ":" in toke:
            try:
                fullkey, val = toke.split(':')
            except ValueError:
                ugh = toke.split(':') # must be colons in the val
                fullkey = ugh[0]
                val = ''.join(ugh[1:])
        else: # might just be continuation of previous key?
            val = " ".join(toke.split())
            if key is not None and val != "<none>":
                fields[key].append(val)
            continue
        if val == "<none>":
            continue
        keypieces = fullkey.split('_')
        if len(keypieces) == 1:
            key = fullkey
        else: # TODO N.B. this seems kinda wrong; what if it's a multi-word key but just one of them
            keynum = keypieces[-1]
            key = '_'.join(keypieces[:-1])
        if key in fields:
            fields[key].append(val) # assuming keys are ordered...
        else:
            fields[key] = [val]
    return fields

def isint(s):
    try:
        num = int(s)
        return True
    except ValueError:
        return False

def get_wikibio_fields(tokes):
    """
    key -> list of words
    """
    fields = OrderedDict()
    key = None # remembers last key
    for toke in tokes:
        if ":" in toke:
            try:
                fullkey, val = toke.split(':')
            except ValueError:
                ugh = toke.split(':') # must be colons in the val
                fullkey = ugh[0]
                val = ''.join(ugh[1:])
        else: # might just be continuation of previous key?
            val = " ".join(toke.split())
            if key is not None and val != "<none>":
                fields[key].append(val)
            continue
        # if val == "<none>":
        #     continue
        keypieces = fullkey.split('_')
        if len(keypieces) == 1:
            key = fullkey
        else:
            if isint(keypieces[-1]):
                key = '_'.join(keypieces[:-1])
            else:
                key = fullkey
        key += ":" # to distinguish from reg words
        if key in fields and val != "<none>":
            fields[key].append(val) # assuming keys are ordered...
        elif val == "<none>": # make it empty
            fields[key] = []
        else:
            fields[key] = [val]
    return fields

def get_e2e_fields(tokes):
    """
    returns keyname -> list of words dict
    """
    fields = OrderedDict()
    state = None
    for toke in tokes:
        if "__start" in toke:
            assert state is None
            state = toke[len("__start_"):]
        elif "__end" in toke:
            endstate = toke[len("__end_"):]
            assert endstate == state
            state = None
        else:
            assert state is not None
            if state in fields:
                fields[state].append(toke)
            else:
                fields[state] = [toke]

    return fields

def get_e2e_fieldspp(tokes):
    """
    returns keyname -> list of words dict
    """
    ffkey = "familyFriendly__"
    fields = get_e2e_fields(tokes)
    if ffkey in fields:
        fields[ffkey + fields[ffkey][0]] = "1"
        del fields[ffkey]
    return fields

# taken from tgen.futil
def tgen_tokenize(text):
    """Tokenize the given text (i.e., insert spaces around all tokens)"""
    toks = ' ' + text + ' '  # for easier regexes

    # enforce space around all punct
    toks = regex.sub(r'(([^\p{IsAlnum}\s\.\,−\-])\2*)', r' \1 ', toks)  # all punct (except ,-.)
    toks = regex.sub(r'([^\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. & no numbers
    toks = regex.sub(r'([^\p{N}])([,.])([\p{N}])', r'\1 \2 \3', toks)  # ,. preceding numbers
    toks = regex.sub(r'([\p{N}])([,.])([^\p{N}])', r'\1 \2 \3', toks)  # ,. following numbers
    toks = regex.sub(r'(–-)([^\p{N}])', r'\1 \2', toks)  # -/– & no number following
    toks = regex.sub(r'(\p{N} *|[^ ])(-)', r'\1\2 ', toks)  # -/– & preceding number/no-space
    toks = regex.sub(r'([-−])', r' \1', toks)  # -/– : always space before

    # keep apostrophes together with words in most common contractions
    toks = regex.sub(r'([\'’´]) (s|m|d|ll|re|ve)\s', r' \1\2 ', toks)  # I 'm, I 've etc.
    toks = regex.sub(r'(n [\'’´]) (t\s)', r' \1\2 ', toks)  # do n't

    # other contractions, as implemented in Treex
    toks = regex.sub(r' ([Cc])annot\s', r' \1an not ', toks)
    toks = regex.sub(r' ([Dd]) \' ye\s', r' \1\' ye ', toks)
    toks = regex.sub(r' ([Gg])imme\s', r' \1im me ', toks)
    toks = regex.sub(r' ([Gg])onna\s', r' \1on na ', toks)
    toks = regex.sub(r' ([Gg])otta\s', r' \1ot ta ', toks)
    toks = regex.sub(r' ([Ll])emme\s', r' \1em me ', toks)
    toks = regex.sub(r' ([Mm])ore\'n\s', r' \1ore \'n ', toks)
    toks = regex.sub(r' \' ([Tt])is\s', r' \'\1 is ', toks)
    toks = regex.sub(r' \' ([Tt])was\s', r' \'\1 was ', toks)
    toks = regex.sub(r' ([Ww])anna\s', r' \1an na ', toks)

    # clean extra space
    toks = regex.sub(r'\s+', ' ', toks)
    toks = toks.strip()
    return toks
