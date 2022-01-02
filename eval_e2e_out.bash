python strip_tags_and_deriv.py < $1 > $1.strpd
python postprocess.py $1.strpd $1.pp # postprocess.py is from https://github.com/UFAL-DSG/tgen/blob/master/e2e-challenge/postprocess/postprocess.py
python ../e2e-metrics/measure_scores.py -p $2 $1.pp

