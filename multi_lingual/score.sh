RUN=$1
PATH=$2
Model_idx=$3
src=en
key=$1_$3

tgt=cs
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=fr
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=de
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=gu
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=ja
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=my
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=ro
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=ru
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=vi
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}
tgt=zh
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key}

