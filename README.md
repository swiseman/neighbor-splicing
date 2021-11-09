Code for [Data-to-text Generation by Splicing Together Nearest Neighbors](https://arxiv.org/pdf/2101.08248.pdf).


### Requirements
- pytorch 1.6.0
- transformers 3.3.1

### Making the E2E data
The `data/e2e` directory already contains the [E2E data](https://github.com/tuetschek/e2e-dataset) (Novikova et al., 2017) in a pre-processed state. To replicate this preprocessing do the following:

- Retrieve neighbors with the following commands. These write neighbors to `data/e2e/{train,val}-nes.txt`.
```
python -u writenes.py -ne_fi data/e2e/train-src.txt -train_tgt_fi data/e2e/train-tgt.txt -out_fi data/e2e/train-nes.txt -nne 100 -cuda -e2e
```

```
python -u writenes.py -ne_fi data/e2e/train-src.txt -train_tgt_fi data/e2e/train-tgt.txt -val_src_fi data/e2e/val-src.txt -out_fi data/e2e/val-nes.txt -nne 100 -cuda -e2e
```

```
python -u writenes.py -ne_fi data/e2e/train-src.txt -train_tgt_fi data/e2e/train-tgt.txt -val_src_fi data/e2e/test-src.txt -out_fi data/e2e/test-nes.txt -nne 100 -cuda -e2e
```

- Before calculating the derivations from neighbors we mask out tokens that appear in both the source table and the target text sequence:
```
python -u mask_tgts.py -data data/e2e/ -src_fi data/e2e/train-src.txt -tgt_fi data/e2e/train-tgt.txt -e2e -split_dashes > data/e2e/masked-train-tgt.txt
```


- Finally calculate derivations:
```
python -u gold_derivs.py -val_src_fi data/e2e/train-src.txt -val_tgt_fi data/e2e/train-tgt.txt -val_ne_fi data/e2e/train-nes.txt -ne_tgt_fi data/e2e/masked-train-tgt.txt -out_fi data/e2e/train-encl-derivs.dat -split_dashes -nne 20 -e2e -max_srclen 100 -max_tgtlen 70 -enclose
```

```
python -u gold_derivs.py -val_src_fi data/e2e/val-src.txt -val_tgt_fi data/e2e/val-tgt.txt -val_ne_fi data/e2e/val-nes.txt -ne_tgt_fi data/e2e/masked-train-tgt.txt -out_fi data/e2e/val-encl-derivs.dat -split_dashes -nne 20 -e2e -max_srclen 100 -max_tgtlen 70 -enclose -val
```

### Making the WikiBio data
The WikiBio data (Lebret et al., 2016) can be downloaded [here](https://rlebret.github.io/wikipedia-biography-dataset/). Follow the instructions at the preceding link to obtain aligned source (i.e., `*.box`) and target files. (You can also get the data from [HuggingFace](https://huggingface.co/datasets/wiki_bio)). Call the `*.box` files `{train|val|test}-src.txt` and the target files `{train|val|test}-tgt.txt` and put them in the same directory.

Below we detail the commands necessary for calculating nearest neighbors and oracle derivations. **These files can also be downloaded from this [folder](https://drive.google.com/drive/folders/1xtO5_yyjZWOLHcY5Wyr6fIUZrC1bq0BV?usp=sharing).**


- Retrieve neighbors (you may want to increase or decrease `-bsz` in the script below depending your hardware):
```
python -u writenes.py -ne_fi data/wb/train-src.txt -train_tgt_fi data/wb/train-tgt.txt -out_fi data/wb/train-nes.txt -nne 40 -cuda
```

```
python -u writenes.py -ne_fi data/wb/train-src.txt -train_tgt_fi data/wb/train-tgt.txt -val_src_fi data/wb/val-src.txt -out_fi data/wb/val-nes.txt -nne 40 -cuda
```

```
python -u writenes.py -ne_fi data/wb/train-src.txt -train_tgt_fi data/wb/train-tgt.txt -val_src_fi data/wb/test-src.txt -out_fi data/wb/test-nes.txt -nne 40 -cuda
```

- Mask target tokens appearing in the source table:
```
python -u mask_tgts.py -data data/wb/ -src_fi data/wb/train-src.txt -tgt_fi data/wb/train-tgt.txt -split_dashes > data/wb/masked-train-tgt.txt
```

- Calculate derivations. This can also be parallelized using the `-wrkr` option (e.g., 2 processes might use arguments `-wrkr 1,2`, `-wrkr 2,2` respectively) and then all the separately produced derivations can be collected with `collect_derivations.py`

```
python -u gold_derivs.py -val_src_fi data/wb/train-src.txt -val_tgt_fi data/wb/train-tgt.txt -val_ne_fi data/wb/train-nes.txt -ne_tgt_fi data/wb/masked-train-tgt.txt -out_fi data/wb/train-encl-derivs.dat -split_dashes -nne 20 -max_srclen 130 -max_tgtlen 50 -enclose
```

```
python -u gold_derivs.py -val_src_fi data/wb/val-src.txt -val_tgt_fi data/wb/val-tgt.txt -val_ne_fi data/wb/val-nes.txt -ne_tgt_fi data/wb/masked-train-tgt.txt -out_fi data/wb/val-encl-derivs.dat -split_dashes -nne 20 -max_srclen 130 -max_tgtlen 50 -enclose -val
```

### Training

```
python -u train.py -cuda -data data/e2e -adamhyps 0.9,0.999,1e-7,0.001 -Topts CNSWx2 -embdim 420 -enc_layers 6 -epochs 100 -ffdim 650 -init 0.01 -lr 0.001 -min_nes 20 -min_seq_accum 400 -min_valnes 20 -nheads 7 -nne 20 -optalg adamw -seed 3636 -senc_layers 6 -share_encs -src_mode mask -val_bsz 16 -val_mbs_per_epoch 300 -bsz 8 -mbs_per_epoch 5260 -log_interval 1000 -flat_moves -vocopts 20 20 -1 -1 -drop 0.1 -warmup_steps 4000 -split_dashes -enclose -save my_e2e_model.pt
```

```
python -u train.py -cuda -data data/wb -adamhyps 0.9,0.999,1e-7,0.001 -Topts CNSWx2 -embdim 420 -enc_layers 6 -epochs 100 -ffdim 650 -init 0.01 -lr 0.0005 -min_nes 20 -min_seq_accum 400 -min_valnes 20 -nheads 7 -nne 20 -optalg adamw -seed 3636 -senc_layers 6 -share_encs -src_mode mask -val_bsz 16 -val_mbs_per_epoch 850 -bsz 6 -mbs_per_epoch 48000 -log_interval 1000 -flat_moves -vocopts 50 50 -1 -1 -drop 0.1 -warmup_steps 4000 -split_dashes -enclose -save my_wb_model.pt
```

### Generation

```
python -u search.py -cuda -data data/e2e/ -val_src_fi src_uniq_valid.txt -val_nefi uniqval-nes.txt -get_trace -bsz 1 -train_from my_e2e_model.pt-l -split_dashes -K 5 -max_moves 16 -nne 20 -out_fi e2e_val.out
```

```
python -u search.py -cuda -data data/wb/ -val_src_fi val-src.txt -val_nefi val-nes.txt -get_trace -bsz 1 -train_from my_wb_model.pt-l -split_dashes -K 10 -max_moves 28 -nne 20 -out_fi wb_val.out
```
