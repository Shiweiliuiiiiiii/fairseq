import torch
import os 
import numpy as np 
import matplotlib.pyplot as plt
import math 
import sys


def read_bleu(path):
    with open(path) as f:
        data = f.readlines()
    data = data[-1]
    key = 'Generate test with beam=5: BLEU4 = '
    bleu_score = data[len(key): len(key)+5]
    bleu_score = float(bleu_score)
    return bleu_score


lang_list = ['fr', 'cs', 'de', 'gu', 'ja', 'my', 'ro', 'ru', 'vi', 'zh']
sparsity = 0
task = '_1010_'

for lang in lang_list:
    path = 'en_' + lang + task + str(sparsity) + '.txt'
    bleu = read_bleu(os.path.join('bleu_result', path))
    print(task, sparsity, lang, bleu)












