import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task
import os, re

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

check_point_folder = '/projects/0/prjste21060/projects/pruning_fails/QA/robert/commonsenseqa/'
model_files = os.listdir('/projects/0/prjste21060/projects/pruning_fails/QA/robert/commonsenseqa/')
model_files = sorted_nicely(model_files)

for file in model_files:

    roberta = RobertaModel.from_pretrained(check_point_folder+str(file), 'checkpoint_best.pt', 'data/CommonsenseQA')

    total_size = 0
    sparse_size = 0
    # for module in self.modules:
    for name, weight in roberta.named_parameters():
        dense_weight_num = weight.numel()
        sparse_weight_num = (weight != 0).sum().int().item()
        total_size += dense_weight_num
        sparse_size += sparse_weight_num
        layer_density = sparse_weight_num / dense_weight_num
        print(f'sparsity of layer {name} with tensor {weight.size()} is {1 - layer_density}')
    print('Final sparsity level of {0}: {1}'.format(0.6, 1 - sparse_size / total_size))

    roberta.eval()  # disable dropout
    roberta.cuda()  # use the GPU (optional)
    nsamples, ncorrect = 0, 0

    with open('/home/sliu/Projects/fairseq/data/CommonsenseQA/valid.jsonl') as h:
        for line in h:
            example = json.loads(line)
            scores = []
            for choice in example['question']['choices']:
                input = roberta.encode(
                    'Q: ' + example['question']['stem'],
                    'A: ' + choice['text'],
                    no_separator=True
                )
                score = roberta.predict('sentence_classification_head', input, return_logits=True)
                scores.append(score)
            pred = torch.cat(scores).argmax()
            answer = ord(example['answerKey']) - ord('A')
            nsamples += 1
            if pred == answer:
                ncorrect += 1

    print('Accuracy: ' + str(ncorrect / float(nsamples)))