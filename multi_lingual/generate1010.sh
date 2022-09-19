model=$1
source_lang=$2
target_lang=$3
path_2_data=examples/multilingual/multidata
lang_list=examples/multilingual/lang_list.txt
lang_pairs=en-fr,en-cs,en-de,en-gu,en-ja,en-my,en-ro,en-ru,en-vi,en-zh,zh-en,vi-en,ru-en,ro-en,my-en,ja-en,gu-en,de-en,cs-en,fr-en
key=$4

CUDA_VISIBLE_DEVICES=$5 fairseq-generate $path_2_data \
    --path $model \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $source_lang \
    --target-lang $target_lang \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 32 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}_${key}.txt