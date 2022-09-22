cd examples/multilingual
wget https://www.dropbox.com/sh/w25c7ov341duugr/AADJ22QSfA_v7LNZNBrImzlua/multilingual_translation/data.tar?dl=0
tar -xvf data.tar 
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
tar -xzvf mbart.cc25.v2.tar.gz
cd ../../