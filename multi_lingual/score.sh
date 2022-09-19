PATH=$1
Model_idx=$2
src=en
key=mling22_$2

tgt=cs
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=fr
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=de
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=gu
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=ja
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=my
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=ro
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=ru
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=vi
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=zh
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/generate22.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}


