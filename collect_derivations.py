import os
import pickle
import argparse


def dump(all_derivs, fi):
    with open(fi, "ab") as f:
        for thing in all_derivs:
            pickle.dump(thing, f, pickle.DEFAULT_PROTOCOL)

def collect(dirname, fis):
    all_derivs = []
    for fi in fis:
        with open(os.path.join(dirname, fi), "rb") as f:
            while True:
                try:
                    all_derivs.append(pickle.load(f))
                except EOFError:
                    break
    return all_derivs

parser = argparse.ArgumentParser()
parser.add_argument('-prefix', type=str, default="data/e2e/ence2e-n100.dat-", help='')


if __name__ == "__main__":
    args = parser.parse_args()
    dirname, basename = os.path.split(args.prefix)
    fis = sorted((fi for fi in os.listdir(dirname) if fi.startswith(basename)),
                 key=lambda x: int(x[len(basename):]))
    assert len(fis) > 0
    print("collecting", len(fis))
    print("ordered files:", fis)
    all_derivs = collect(dirname, fis)
    print("got", len(all_derivs), "derivations in total")
    outfi = args.prefix[:-1]
    print("dumping to", outfi)
    dump(all_derivs, outfi)
