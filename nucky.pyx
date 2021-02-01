from collections import defaultdict, OrderedDict
cimport numpy as np
import numpy as np

def make_database(neighbs):
    cdef int i, j, n, N, N_i
    subs = defaultdict(list)
    N = len(neighbs)
    #for n, ne in enumerate(neighbs):
    for n in range(N):
        ne = neighbs[n]
        N_i = len(ne)
        for i in range(N_i):
        #for i in range(len(ne)):
            for j in range(i+1, N_i+1):
                subs[tuple(ne[i:j])].append((n, i, j))
    return subs


# this assumes nts are ints and maps back to string
def backtrack(nt, neighbs, bps, l, r):
    cdef int cX = -1
    cdef int cR = -2
    cdef int cS = -3
    cdef int cC = -4
    cdef dict tostrnt = {-1: 'X', -2: 'R', -3: 'S', -4: 'C', -5: 'S0'}
    stuff = None
    for nt1, stuff1 in bps[l][r]:
        if nt1 == nt:
            stuff = stuff1
            break
    assert stuff is not None
    if stuff[0] == cX and not isinstance(stuff[1], tuple):
        return ('X',) + stuff[1:]
    elif stuff[0] == cR and not isinstance(stuff[1], tuple):
        return ('R',) + stuff[1:]
    elif nt == (cS,) and not isinstance(stuff[1], tuple):
        return ('S',) + stuff # stuff is (m, l, r)
    #print(nt, "stuff is", stuff)
    k, leftnt, rightnt = stuff
    left = backtrack(leftnt, neighbs, bps, l, k)
    right = backtrack(rightnt, neighbs, bps, k, r)
    return [(tostrnt[nt[0]],) + nt[1:], left, right]


# about 15x faster than python implementation
def parse(x, neighbs, maxx_sz):
    cdef int i, j, k, N, length, scost, rcost, ccost, casint
    cdef int l, r, lnt, lm, ll, lr, lc, rnt, rm, rl, rr, rc
    cdef int imaxx_sz = maxx_sz
    cdef int cX = -1
    cdef int cR = -2
    cdef int cS = -3
    cdef int cC = -4
    N = len(x)
    # just guessing how many times a thing could be repeated...
    table = np.zeros((N, N+1, imaxx_sz, 5), dtype=np.dtype('i'))
    tlens = np.zeros((N, N+1), dtype=np.dtype('i'))
    # memory views or whatever
    cdef int [:, :, :, :] tablev = table
    cdef int [:, :] tlensv = tlens
    cdef list bps = [] # making this a defaultdict is very slightly slower
    subs = make_database(neighbs)
    # base stuff
    #for i in range(len(x)):
    for i in range(N):
        bps.append([])
        # make dummy stuff
        for j in range(N+1):
            bps[i].append([])
        #for j in range(i+1, len(x)+1):
        for j in range(i+1, N+1):
            key = tuple(x[i:j])
            if key in subs:
                # everything can either be a regular NT or a special one
                for (m, l, r) in subs[key]:
                    # fmt is char, m, l, r, cost
                    tablev[i][j][tlensv[i][j]][0] = cX; tablev[i][j][tlensv[i][j]][1] = m
                    tablev[i][j][tlensv[i][j]][2] = l; tablev[i][j][tlensv[i][j]][3] = r
                    tlensv[i][j] += 1
                    #table[i][j].append((('X', m, l, r), 0))
                    bps[i][j].append(((cX, m, l, r), (cX, m, l, r)))
                    # do a unary R -> expansion
                    tablev[i][j][tlensv[i][j]][0] = cR; tablev[i][j][tlensv[i][j]][1] = m
                    tablev[i][j][tlensv[i][j]][2] = l
                    tlensv[i][j] += 1
                    #table[i][j].append((('R', m, l), 0))
                    bps[i][j].append(((cR, m, l), (cR, m, l, r)))
                # do a unary S -> expansion
                tablev[i][j][tlensv[i][j]][0] = cS
                tablev[i][j][tlensv[i][j]][4] = 1
                tlensv[i][j] += 1
                #table[i][j].append((('S',), 1))
                bps[i][j].append(((cS,), subs[key][0])) # just take the first; doesn't matter
            elif j == i+1: # can't generate a word
                assert False
    # all the substring matches are already there
    skey = (cS,)
    #for length in range(2, len(x)+1):
    for length in range(2, N+1):
        #for i in range(len(x)-length+1):
        for i in range(N-length+1):
            # if (i, i+length) in table: # nothing to do; we already have optimal tiling
            #     continue
            if tlensv[i][i+length] > 0:
                continue
            #bests = {}
            bests = OrderedDict() # actually {} should be ordered too
            for k in range(i+1, i+length):
                #if (i, k) in table and (k, i+length) in table:
                if tlensv[i][k] > 0 and tlensv[k][i+length] > 0:
                    for l in range(tlensv[i][k]):
                        lnt = tablev[i][k][l][0]; lm = tablev[i][k][l][1]; ll = tablev[i][k][l][2]
                        lr = tablev[i][k][l][3]; lc = tablev[i][k][l][4]
                        for r in range(tlensv[k][i+length]):
                            rnt = tablev[k][i+length][r][0]; rm = tablev[k][i+length][r][1]
                            rl = tablev[k][i+length][r][2]; rr = tablev[k][i+length][r][3]
                            rc = tablev[k][i+length][r][4]
                    # for lnt, lc in table[i][k]:
                    #     for rnt, rc in table[k][i+length]:
                            if lnt == cX and rnt == cC and lm == rm:
                            #if lnt[0] == 'X' and rnt[0] == 'C' and lnt[1] == rnt[1]:
                                if rl >= lr:
                                #if rnt[2] >= lnt[3]: # C's l >= X's r
                                    scost = lc + rc + 1 # S -> X^m_:r C^m_s>=r
                                    if skey not in bests or scost < bests[skey][0]:
                                        bests[skey] = (scost, k, (cX, lm, ll, lr), (cC, rm, rl))
                                        #bests[skey] = (scost, k, lnt, rnt)
                                # if rnt[2] >= lnt[3]: # C's l >= X's r
                                    rcost = lc + rc # R^m_s -> X^m_s:r C^m_t>=r
                                    rkey = (cR, lm, ll)
                                    #rkey = ('R', lnt[1], lnt[2]) # (R, m, l)
                                    if rkey not in bests or rcost < bests[rkey][0]:
                                        bests[rkey] = (rcost, k, (cX, lm, ll, lr), (cC, rm, rl))
                                        #bests[rkey] = (rcost, k, lnt, rnt)
                            elif lnt == cS:
                            #elif lnt[0] == 'S':
                                if rnt == cS:
                                #if rnt[0] == 'S':
                                    scost = lc + rc
                                    if skey not in bests or scost < bests[skey][0]:
                                        bests[skey] = (scost, k, skey, skey)
                                        #bests[skey] = (scost, k, lnt, rnt)
                                elif rnt == cR:
                                #elif rnt[0] == 'R':
                                    ccost = lc + rc
                                    ckey = (cC, rm, rl) # (C, m, l)
                                    #ckey = ('C', rnt[1], rnt[2]) # (C, m, l)
                                    if ckey not in bests or ccost < bests[ckey][0]:
                                        bests[ckey] = (ccost, k, skey, (cR, rm, rl))
                                        #bests[ckey] = (ccost, k, lnt, rnt)

            for key, rest in bests.items():
                tablev[i][i+length][tlensv[i][i+length]][0] = key[0]
                if tablev[i][i+length][tlensv[i][i+length]][0] != cS:
                    tablev[i][i+length][tlensv[i][i+length]][1] = key[1]
                    tablev[i][i+length][tlensv[i][i+length]][2] = key[2]
                if tablev[i][i+length][tlensv[i][i+length]][0] == cX:
                    tablev[i][i+length][tlensv[i][i+length]][3] = key[3]
                tablev[i][i+length][tlensv[i][i+length]][4] = rest[0]
                tlensv[i][i+length] += 1
                #table[i][i+length].append((key, rest[0]))
                bps[i][i+length].append((key, rest[1:]))

    return table, bps, subs

#### Below functions should be identical to ones in cky; just duplicating so we only need 1 file

def movesfromtree(tree, movelist):
    if tree[0] != ('S',) and not isinstance(tree[1], list):
        movelist.append(tree[:-1])
    else:
        if tree[0][0] == 'X':
            movelist.append(tree[0][:-1])
        for child in tree[1]:
            movesfromtree(child, movelist)


def postproc_nbtree(tree, curri, earliest=True):
    """
    consumes outputs of fixtree; removes R's etc
    if earliest does earliest replace, which should behave the same as the initial impl
    returns tree where nodes have format (NT, neidx, l, r, i, skip, ntokens_in_subtree)
    """
    if isinstance(tree, tuple):
        #nt, neidx, l, r = tree
        tree = list(tree) + [curri, 0] # for insloc, skip
        tree.append(tree[3] - tree[2]) # for subtree size
        return tree
    root, children = tree
    if root[0] == 'X':
        #nt, neidx, l, r = root
        nuroot = list(root) + [curri, 0]
        lastr = nuroot[3]
        size = (nuroot[3] - nuroot[2]) # size of this subtree so far
        curri += size # increment idx from left with size so far
    else:
        assert len(root) == 1 # should only be an S if topmost one
        nuroot = root
        size = 0

    nuchildren, repidx = [], -1
    for child in children:
        ppchild = postproc_nbtree(child, curri, earliest=earliest)
        csize = ppchild[-1] if len(ppchild) == 7 else ppchild[0][-1]
        curri += csize
        size += csize
        if len(ppchild) == 7 and ppchild[0] == 'R': # an R leaf
            _, cneidx, cl, cr, _, _, _ = ppchild
            assert cneidx == nuroot[1]
            skip = cl - lastr
            assert repidx != -1
            if len(nuchildren[repidx]) == 7: # tree doing replace is a leaf
                nuchildren[repidx][5] = skip
            else: # a tree
                nuchildren[repidx][0][5] = skip
            lastr = cr
            repidx = -1
        else: # only ignore R children
            nuchildren.append(ppchild)
            if repidx == -1 or not earliest:
                repidx = len(nuchildren) - 1
    if root[0] == 'X': # update lastr
        nuroot[3] = lastr
        nuroot.append(size)
    return [nuroot, nuchildren]


# Rule1: if S -> Y C, make it Y -> C's children
# Rule2: collapse everything else
def fixtree(tree):
    """
    accepts a binary tree but returns nonbinary one
    """
    if isinstance(tree, tuple): # a leaf
        return tree # or maybe [tree, []]
    root, left, right = tree
    if root[0] == 'S' and right[0][0] == 'C': # S -> X C
        assert left[0] == 'X'
        # connect C's left and right children to X
        cltree = fixtree(right[1]) # an S
        crtree = fixtree(right[2]) # an R
        nuchildren = []
        if isinstance(right[1], tuple) or cltree[0][0] == 'X': # an S-leaf
            nuchildren.append(cltree) # cltree needs to be an S thing
        else: # an S-tree, so collapse up its children
            nuchildren.extend(cltree[1])
        if isinstance(right[2], tuple): # it's an R-leaf, so just append
            nuchildren.append(crtree)
        else: # it has its own children, which we want to collapse
            nuchildren.extend(crtree[1])
        # now we have X -> C's children
        xtree = [left, nuchildren] # might wanna do tuple(left)?
        return xtree
    # otherwise I think we just always collapse?
    nuchildren = []
    ltree = fixtree(left)
    if isinstance(left, tuple) or ltree[0][0] == 'X':
        assert left[0] in ['X', 'S', ('S',)] # will be ('S',) if ltree[0][0] == 'X'
        if root[0] == 'R': # change NT to R so we know it's not actually put in now
            ltree = ('R',) + ltree[1:]
        nuchildren.append(ltree)
    else:
        #assert left[0][0] == 'S' # i think we can merge
        assert ltree[0][0] == 'S'
        nuchildren.extend(ltree[1])
    rtree = fixtree(right)
    if isinstance(right, tuple) or rtree[0][0] == 'X': # a tree headed by X
        nuchildren.append(rtree)
    else: # merge
        nuchildren.extend(rtree[1])
    return [root, nuchildren]


def get_movetree(tree):
    ftree = fixtree(tree)
    if ftree[0][0] != 'S': # root must've been an S -> X C
        assert ftree[0][0] == 'X'
        ftree = [('S',), [ftree]]
    elif isinstance(ftree[0], str): # just one insert
        ftree = [('S',), [ftree]]
    return postproc_nbtree(ftree, 0)


# gets leaves in format [nt, neidx, l, r, skip, finalr]
def read_tree(tree):
    if isinstance(tree, tuple):
        ltree = list(tree)
        ltree.append(0) # no skip
        return [ltree]
    # otherwise should be a list
    root, left, right = tree
    if isinstance(right, list) and right[0][0] == 'C':
        assert root[0] in ['S', 'R', 'S0']
        assert left[0] == 'X' and (root[0] in ['S', 'S0'] or left[1] == root[1])
        # get left subtree, which should be a terminal
        left = list(left)
        left.append(0) # no skip by default
        # get left and right descendants of right branch
        clmoves, crmoves = read_tree(right)
        # crmoves[0] (i.e., leftmost child of crmoves) should be an X/R
        crleftmost = crmoves[0]
        # neidx and l should agree
        assert crleftmost[1] == right[0][1] and right[0][2] == crleftmost[2]
        # check for replace; if so update left subtree of C
        skip = crleftmost[2] - left[3] # if crleftmost.l > left.r it's a replace
        if skip > 0: # make the first child of clmoves do the replace
            clmoves[0][4] = skip
        # finally update left subtree (a terminal) w/ finalr or to be skipped
        if root[0] in ['S', 'S0']: # need finalr
            assert left[0] == 'X' and left[1] == right[0][1]
            assert crmoves[-1][0] == 'R' and crmoves[-1][1] == right[0][1]
            # update initial insert
            finalr = crmoves[-1][3] # final r from all the way down the tree
            left.append(finalr)
        else: # root is R, so make sure we skip it
            left[0] = 'R' # so we know to skip
        return [left] + clmoves + crmoves
    elif root[0] == 'C':
        leftmoves = read_tree(left)
        rightmoves = read_tree(right)
        return leftmoves, rightmoves # keep them separate
    else:
        leftmoves = read_tree(left)
        rightmoves = read_tree(right)
        return leftmoves + rightmoves


# gets goldish moves
# format is [action, neidx, l, r, curri, skip]
def get_moves(tree, leaves=None):
    if leaves is None:
        leaves = read_tree(tree)
    moves = []
    curri = 0
    for leaf in leaves:
        nt, neidx, l, r = leaf[0:4]
        # skip is 0 by default
        move = ["insert", neidx, l, r, curri, 0]
        if leaf[4] > 0: # a replace
            move[0] = "replace"
            move[5] = leaf[4]
        # increment curri
        curri += (r - l)
        if len(leaf) > 5: # fix up r
            move[3] = leaf[5]
        if nt != 'R':
            moves.append(move)
    return moves


def reconstruct(moves, neighbs):
    canvas = []
    for move in moves:
        neidx, l, r, ii, skip = move[1:]
        canvas = canvas[:ii] + neighbs[neidx][l:r] + canvas[ii+skip:]
    return canvas


def greedy_tag(x, neighbs):
    cdef int i, j, k, N, M
    cdef int l, r, m
    N = len(x)
    cdef list sames
    cdef list news
    cdef list moves = []
    subs = make_database(neighbs)
    #print("made db")
    cdef set used = set()
    i = 0
    while i < N:
        #print("oy", i)
        for j in range(N, i, -1):
            #print("ey", j)
            key = tuple(x[i:j])
            if key in subs: # break ties by whether we've used this neighbor before
                sames = []
                news = []
                M = len(subs[key])
                for k in range(M):
                    if subs[key][k][0] in used:
                        sames.append(k)
                    else:
                        news.append(k)
                if len(sames) > 0:
                    ridx = np.random.randint(len(sames))
                    m, l, r = subs[key][sames[ridx]]
                else:
                    ridx = np.random.randint(len(news))
                    m, l, r = subs[key][news[ridx]]
                used.add(m)
                moves.append(["insert", m, l, r, i, 0])
                i += (r - l)
                break
    return moves
