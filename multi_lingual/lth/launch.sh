# example
nohup bash multi_lingual/lth/imp_2to2.sh 0 > log_LTH_2to2.out 2>&1 &
nohup bash multi_lingual/lth/imp_5to5.sh 1 > log_LTH_5to5.out 2>&1 &
nohup bash multi_lingual/lth/imp_10to1.sh 2 > log_LTH_10to1.out 2>&1 &
nohup bash multi_lingual/lth/imp_10to10.sh 3 > log_LTH_10to10.out 2>&1 &

