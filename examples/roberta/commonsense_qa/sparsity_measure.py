import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task
import os, re


import sys

removed_layers = ['in_proj_weight', 'out_proj_weight', 'fc1_weight', 'fc2_weight', 'lm_head.dense.weight']
snns = ['gm', 'gm_after',  'gmp', 'IMP', 'random', 'random_after',  'snip']

sparsities = ['0.36', '0.672', '0.8325']

sparsity_all = []
for snn in snns:
    snn_path = os.path.join(sys.argv[1], snn)
    for sparsity in sparsities:

        roberta = RobertaModel.from_pretrained(os.path.join(snn_path, sparsity), 'checkpoint_best.pt', 'data/CommonsenseQA')

        total_zero = 0
        total_weight = 0

        for name, weight in roberta.named_parameters():
            if len(weight.size()) == 2 or len(weight.size()) == 4:
                if name in removed_layers: continue
                sparsity_all.append((weight==0).sum().item() / weight.numel())




