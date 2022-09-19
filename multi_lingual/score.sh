FILE_NAME=$1
PATH=$2
Model_idx=$3
src=en
key=$1_$3

tgt=cs
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=fr
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=de
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=gu
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=ja
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=my
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=ro
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=ru
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=vi
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
tgt=zh
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}
bash multi_lingual/${FILE_NAME}.sh ${PATH}/${Model_idx}/checkpoint_best_iter0.pt ${src} ${tgt} ${key}


