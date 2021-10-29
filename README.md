Code for [Data-to-text Generation by Splicing Together Nearest Neighbors](https://arxiv.org/pdf/2101.08248.pdf).

We show how to set up and train on the [E2E data](https://github.com/tuetschek/e2e-dataset) (Novikova et al., 2017), which we include. The [WikiBio data](https://rlebret.github.io/wikipedia-biography-dataset/) (Lebret et al., 2016) can be downloaded from the preceding link and processed analogously.

## Requirements
- pytorch 1.6.0
- transformers 3.3.1

## Making the E2E data

```
python -u writenes.py -ne_fi data/e2e/train-src.txt -train_tgt_fi data/e2e/train-tgt.txt -out_fi data/e2e/train-nes.txt -nne 100 -cuda -e2e
```

```
python -u writenes.py -ne_fi data/e2e/train-src.txt -train_tgt_fi data/e2e/train-tgt.txt -val_src_fi data/e2e/val-src.txt -out_fi data/e2e/val-nes.txt -nne 100 -cuda -e2e
```

- Mask target tokens
```
python -u mask_tgts.py -data data/e2e/ -src_fi data/e2e/train-src.txt -tgt_fi data/e2e/train-tgt.txt -e2e -split_dashes > data/e2e/masked-train-tgt.txt
```

- Get derivations
```
python -u gold_derivs.py -val_src_fi data/e2e/train-src.txt -val_tgt_fi data/e2e/train-tgt.txt -val_ne_fi data/e2e/train-nes.txt -ne_tgt_fi data/e2e/masked-train-tgt.txt -out_fi data/e2e/encle2e-n20.dat -split_dashes -nne 20 -e2e -max_srclen 100 -max_tgtlen 70 -enclose
```

- Train
```
python -u train.py -cuda -data data/e2e -adamhyps 0.9,0.999,1e-7,0.001 -Topts CNSWx2 -embdim 420 -enc_layers 6 -epochs 100 -ffdim 650 -init 0.01 -lr 0.001 -min_nes 20 -min_seq_accum 400 -min_valnes 20 -nheads 7 -nne 20 -optalg adamw -seed 3636 -senc_layers 6 -share_encs -src_mode mask -val_bsz 16 -val_mbs_per_epoch 300 -bsz 8 -mbs_per_epoch 5260 -log_interval 1000 -flat_moves -vocopts 20 20 -1 -1 -drop 0.1 -warmup_steps 4000 -split_dashes -enclose -save mymodel.pt
```

- Predict

```
python -u search.py -cuda -data data/e2e/ -val_src_fi src_uniq_valid.txt -val_nefi uniqval-nes.txt -get_trace -bsz 1 -train_from mymodel.pt-l -split_dashes -K 5 -max_moves 16 -nne 20 -out_fi e2e_val.out
```
