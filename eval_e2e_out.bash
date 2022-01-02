python strip_tags_and_deriv.py < $1 > $1.strpd
python postprocess.py $1.strpd $1.pp
python ../e2e-metrics/measure_scores.py -p $2 $1.pp

