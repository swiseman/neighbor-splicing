import os
import re

import torch

from utils import get_e2e_fields, get_wikibio_fields

def is_multi_sentence(line):
    """
    checks if there's a period with a capitalized word following it
    """
    # some of the train sentences don't capitalize first letter, so could also do
    # return re.search(r'\s+\.\s+\w', line) is not None
    return re.search(r'\s+\.\s+[A-Z]', line) is not None


class E2ERestrictor:
    """
    dumbest implementation; just keeps a list of kosher neighbor idxs
    """
    def __init__(self, args, multi_sentence=True, yes=True):
        self.keepers = set()
        assert multi_sentence

        with open(os.path.join(args.data, "train-tgt.txt")) as tgtfi:
            for i, tgtline in enumerate(tgtfi):
                akeeper = is_multi_sentence(tgtline.strip())
                if not yes:
                    akeeper = not akeeper
                if akeeper:
                    self.keepers.add(i)

    def restrict(self, neidxs, db, ii):
        return [neidx for neidx in neidxs if neidx in self.keepers]
