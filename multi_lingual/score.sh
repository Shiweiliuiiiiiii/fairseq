RUN=$1
PATH=$2
Model_idx=$3
src=en
key=$1_$3
GPU=$4

tgt=cs
/bin/bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=fr
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=de
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=gu
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=ja
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=my
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=ro
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=ru
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=vi
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

tgt=zh
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key} ${GPU}
bash multi_lingual/${RUN}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${tgt} ${src} ${key} ${GPU}

