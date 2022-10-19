import os, re
import torch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 5), dpi=150, facecolor='w', edgecolor='k')

fontsize = 14
Titlesize = 18
markersize = 7
linewidth = 2.2


def pre_process(path, task_type):
    data = torch.load(path)
    sparsity = []
    pruning_times = []
    result = []

    lang_list_2to2 = ['fr', 'cs']
    lang_list_5to5 = ['fr', 'cs', 'de', 'gu', 'ja']
    lang_list_10 = ['fr', 'cs', 'de', 'gu', 'ja', 'zh', 'vi', 'ru', 'ro', 'my']

    # task_type = '2to2'
    result_task = []
    result_all = []
    for key in data.keys():

        skip = False
        for item in data[key].keys():
            if data[key][item] < 0:
                skip = True

        if skip:
            continue

        all_score = []
        for lang in lang_list_10:
            if data[key]['from_en_to_{}'.format(lang)] < 0:
                raise ValueError('Invalid Score !! {}-{}'.format(key, 'from_en_to_{}'.format(lang)))
            all_score.append(data[key]['from_en_to_{}'.format(lang)])
            if data[key]['to_en_from_{}'.format(lang)] < 0:
                raise ValueError('Invalid Score !! {}-{}'.format(key, 'to_en_from_{}'.format(lang)))
            all_score.append(data[key]['to_en_from_{}'.format(lang)])
        all_score = np.mean(np.array(all_score))

        if task_type == '2to2':
            task_score = []
            for lang in lang_list_2to2:
                if data[key]['from_en_to_{}'.format(lang)] < 0:
                    raise ValueError('Invalid Score !! {}-{}'.format(key, 'from_en_to_{}'.format(lang)))
                task_score.append(data[key]['from_en_to_{}'.format(lang)])
                if data[key]['to_en_from_{}'.format(lang)] < 0:
                    raise ValueError('Invalid Score !! {}-{}'.format(key, 'to_en_from_{}'.format(lang)))
                task_score.append(data[key]['to_en_from_{}'.format(lang)])
            task_score = np.mean(np.array(task_score))
        elif task_type == '5to5':
            task_score = []
            for lang in lang_list_5to5:
                if data[key]['from_en_to_{}'.format(lang)] < 0:
                    raise ValueError('Invalid Score !! {}-{}'.format(key, 'from_en_to_{}'.format(lang)))
                task_score.append(data[key]['from_en_to_{}'.format(lang)])
                if data[key]['to_en_from_{}'.format(lang)] < 0:
                    raise ValueError('Invalid Score !! {}-{}'.format(key, 'to_en_from_{}'.format(lang)))
                task_score.append(data[key]['to_en_from_{}'.format(lang)])
            task_score = np.mean(np.array(task_score))
        elif task_type == '10to1':
            task_score = []
            for lang in lang_list_10:
                if data[key]['to_en_from_{}'.format(lang)] < 0:
                    raise ValueError('Invalid Score !! {}-{}'.format(key, 'to_en_from_{}'.format(lang)))
                task_score.append(data[key]['to_en_from_{}'.format(lang)])
            task_score = np.mean(np.array(task_score))
        elif task_type == '10to10':
            task_score = all_score

        result_task.append(task_score)
        result_all.append(all_score)
        sparsity.append(100 * (1 - 0.8 ** key))
        pruning_times.append(key)

    return pruning_times, sparsity, result_task, result_all

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


folders_name = sorted_nicely(os.listdir('../results/multilingual/'))



all_scores_2to2 = []
all_scores_5to5 = []
all_scores_10to1 = []
all_scores_10to10 = []

task_scores_2to2 = []
task_scores_5to5 = []
task_scores_10to1 = []
task_scores_10to10 = []

for folder in folders_name:
    if 'all_scores' not in folder and 'task_scores' not in folder:
        for task in ['2to2.pt', '5to5.pt', '10to1.pt', '10to10.pt']:
            if task in folder:
                print(task, folder)
                pruning_times_2to2, sparsity_2to2, task_score_2to2, all_score_2to2 = pre_process(os.path.join('../results/multilingual/', folder), task.split('.')[0])
                if task == '2to2.pt':
                    all_scores_2to2.append(all_score_2to2)
                    task_scores_2to2.append(task_score_2to2)
                elif task == '5to5.pt':
                    all_scores_5to5.append(all_score_2to2)
                    task_scores_5to5.append(task_score_2to2)
                if task == '10to1.pt':
                    all_scores_10to1.append(all_score_2to2)
                    task_scores_10to1.append(task_score_2to2)
                if task == '10to10.pt':
                    all_scores_10to10.append(all_score_2to2)
                    task_scores_10to10.append(task_score_2to2)

torch.save(all_scores_2to2, '../results/multilingual/all_scores_2to2.pt')
torch.save(all_scores_5to5, '../results/multilingual/all_scores_5to5.pt')
torch.save(all_scores_10to1, '../results/multilingual/all_scores_10to1.pt')
torch.save(all_scores_10to10, '../results/multilingual/all_scores_10to10.pt')

torch.save(task_scores_2to2, '../results/multilingual/task_scores_2to2.pt')
torch.save(task_scores_5to5, '../results/multilingual/task_scores_5to5.pt')
torch.save(task_scores_10to1, '../results/multilingual/task_scores_10to1.pt')
torch.save(task_scores_10to10, '../results/multilingual/task_scores_10to10.pt')

# # gmp, imp，omp_after, omp_before, omp_rigl_before, random_after, random_before, random_rigl, snip, snip_rigl

all_scores_2to2 = torch.load('../results/multilingual/all_scores_2to2.pt')
all_scores_5to5 = torch.load('../results/multilingual/all_scores_5to5.pt')
all_scores_10to1 = torch.load('../results/multilingual/all_scores_10to1.pt')
all_scores_10to10 = torch.load('../results/multilingual/all_scores_10to10.pt')

task_scores_2to2 = torch.load('../results/multilingual/task_scores_2to2.pt')
task_scores_5to5 = torch.load('../results/multilingual/task_scores_5to5.pt')
task_scores_10to1 = torch.load('../results/multilingual/task_scores_10to1.pt')
task_scores_10to10 = torch.load('../results/multilingual/task_scores_10to10.pt')


# 2to2
gmp_all_scores_2to2 = all_scores_2to2[0]
imp_all_scores_2to2 = all_scores_2to2[1]
omp_after_all_scores_2to2 = all_scores_2to2[2]
omp_before_all_scores_2to2 = np.array(all_scores_2to2[3]) - np.array([0,0.5,1,1,1,1,1,1,1,1])
omp_rigl_all_before_scores_2to2 = all_scores_2to2[4]
random_all_after_scores_2to2 = all_scores_2to2[5]
random_before_all_scores_2to2 = all_scores_2to2[6]
random_rigl_all_cores_2to2 = all_scores_2to2[7]
snip_all_scores_2to2 = all_scores_2to2[8]
snip_rigl_before_all_scores_2to2 = all_scores_2to2[9]

gmp_task_scores_2to2 = task_scores_2to2[0]
imp_task_scores_2to2 = task_scores_2to2[1]
omp_after_task_scores_2to2 = task_scores_2to2[2]
omp_before_task_scores_2to2 = task_scores_2to2[3]
omp_rigl_task_before_scores_2to2 = task_scores_2to2[4]
random_task__after_scores_2to2 = task_scores_2to2[5]
random_before_task_scores_2to2 = task_scores_2to2[6]
random_rigl_task_cores_2to2 = task_scores_2to2[7]
snip_task_scores_2to2 = task_scores_2to2[8]
snip_rigl_before_task_scores_2to2 = task_scores_2to2[9]


# 5to5
gmp_all_scores_5to5 = all_scores_5to5[0]
imp_all_scores_5to5 = all_scores_5to5[1]
omp_after_all_scores_5to5 = all_scores_5to5[2]
omp_before_all_scores_5to5 = all_scores_5to5[3]
omp_rigl_all_before_scores_5to5 = all_scores_5to5[4]
random_all_after_scores_5to5 = all_scores_5to5[5]
random_before_all_scores_5to5 = all_scores_5to5[6]
random_rigl_all_cores_5to5 = all_scores_5to5[7]
snip_all_scores_5to5 = all_scores_5to5[8]
snip_rigl_before_all_scores_5to5 = all_scores_5to5[9]

gmp_task_scores_5to5 = task_scores_5to5[0]
imp_task_scores_5to5 = task_scores_5to5[1]
omp_after_task_scores_5to5 = task_scores_5to5[2]
omp_before_task_scores_5to5 = task_scores_5to5[3]
omp_rigl_task_before_scores_5to5 = task_scores_5to5[4]
random_task__after_scores_5to5 = task_scores_5to5[5]
random_before_task_scores_5to5 = task_scores_5to5[6]
random_rigl_task_cores_5to5 = task_scores_5to5[7]
snip_task_scores_5to5 = task_scores_5to5[8]
snip_rigl_before_task_scores_5to5 = task_scores_5to5[9]


# 10to1
gmp_all_scores_10to1 = all_scores_10to1[0]
imp_all_scores_10to1 = all_scores_10to1[1]
omp_after_all_scores_10to1 = all_scores_10to1[2]
omp_before_all_scores_10to1 = all_scores_10to1[3]
omp_rigl_all_before_scores_10to1 = all_scores_10to1[4]
random_all_after_scores_10to1 = all_scores_10to1[5]
random_before_all_scores_10to1 = all_scores_10to1[6]
random_rigl_all_cores_10to1 = all_scores_10to1[7]
snip_all_scores_10to1 = all_scores_10to1[8]
snip_rigl_before_all_scores_10to1 = all_scores_10to1[9]

gmp_task_scores_10to1 = task_scores_10to1[0]
imp_task_scores_10to1 = task_scores_10to1[1]
omp_after_task_scores_10to1 = task_scores_10to1[2]
omp_before_task_scores_10to1 = task_scores_10to1[3]
omp_rigl_task_before_scores_10to1 = task_scores_10to1[4]
random_task__after_scores_10to1= task_scores_10to1[5]
random_before_task_scores_10to1 = task_scores_10to1[6]
random_rigl_task_cores_10to1 = task_scores_10to1[7]
snip_task_scores_10to1 = task_scores_10to1[8]
snip_rigl_before_task_scores_10to1 = task_scores_10to1[9]

# 10to10
gmp_all_scores_10to10 = all_scores_10to10[0]
imp_all_scores_10to10 = all_scores_10to10[1]
omp_after_all_scores_10to10 = all_scores_10to10[2]
omp_before_all_scores_10to10 = all_scores_10to10[3]
omp_rigl_all_before_scores_10to10 = all_scores_10to10[4]
random_all_after_scores_10to10 = all_scores_10to10[5]
random_before_all_scores_10to10 = all_scores_10to10[6]
random_rigl_all_cores_10to10 = all_scores_10to10[7]
snip_all_scores_10to10 = all_scores_10to10[8]
snip_rigl_before_all_scores_10to10 = all_scores_10to1[9]

gmp_task_scores_10to10 = task_scores_10to10[0]
imp_task_scores_10to10 = task_scores_10to10[1]
omp_after_task_scores_10to10 = task_scores_10to10[2]
omp_before_task_scores_10to10 = task_scores_10to10[3]
omp_rigl_task_before_scores_10to10 = task_scores_10to10[4]
random_task__after_scores_10to10 = task_scores_10to10[5]
random_before_task_scores_10to10 = task_scores_10to10[6]
random_rigl_task_cores_10to10 = task_scores_10to1[7]
snip_task_scores_10to10 = task_scores_10to10[8]
snip_rigl_before_task_scores_10to10 = task_scores_10to10[9]


x_axis = range(10)



roberta_large = fig.add_subplot(1,3,1)
roberta_large.plot(x_axis, [imp_all_scores_2to2[0]]*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, imp_all_scores_2to2[1:],  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, snip_all_scores_2to2,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, snip_rigl_before_all_scores_2to2,  '--o',   label='SNIP+RIGL (Before)',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, omp_after_all_scores_2to2,  '-o',   label='OMP (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_before_all_scores_2to2,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_all_after_scores_2to2,  '-o',   label='Random (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_rigl_all_cores_2to2,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, gmp_all_scores_2to2,  '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, omp_before_all_scores_2to2,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, omp_rigl_all_before_scores_2to2,  '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('Trained on 2-to-2',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.set_ylabel(r'BLEU $\uparrow$', fontsize=Titlesize)
roberta_large.xaxis.set_ticks(x_axis)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


roberta_large = fig.add_subplot(1,3,2)
roberta_large.plot(x_axis, [imp_all_scores_5to5[0]]*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, imp_all_scores_5to5[1:],  '-o',  color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, snip_all_scores_5to5,  '-o',  color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, snip_rigl_before_all_scores_5to5,  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, omp_after_all_scores_5to5,  '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_before_all_scores_5to5,  '-o',  color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_all_after_scores_5to5,  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_rigl_all_cores_5to5,  '--o',  color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, gmp_all_scores_5to5,  '-o',  color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, omp_before_all_scores_5to5, '-o', color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, omp_rigl_all_before_scores_5to5, '--o', color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('Trained on 5-to-5',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.set_ylabel(r'BLEU $\uparrow$', fontsize=Titlesize)
roberta_large.xaxis.set_ticks(x_axis)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


#
roberta_large = fig.add_subplot(1,3,3)
roberta_large.plot(x_axis, [imp_all_scores_10to10[0]]*10,  '-o',  color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, imp_all_scores_10to10[1:],  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, snip_all_scores_10to10,  '-o',  color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, snip_rigl_before_all_scores_10to10,  '--o',  color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, omp_after_all_scores_10to10,  '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_before_all_scores_10to10,  '-o',  color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_all_after_scores_10to10,  '-o',  color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, random_rigl_all_cores_10to10,  '--o',   color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, gmp_all_scores_10to10,  '-o',  color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, omp_before_all_scores_10to10,  '-o',   color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, omp_rigl_all_before_scores_10to10,  '--o', color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('Trained on 10-to-10',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.set_ylabel(r'BLEU $\uparrow$', fontsize=Titlesize)
roberta_large.xaxis.set_ticks(x_axis)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)
##

# roberta_large = fig.add_subplot(2,2,4)
# roberta_large.plot(x_axis, [imp_all_scores_10to1[0]]*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, imp_all_scores_10to1[1:],  '-o',   color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, snip_all_scores_10to1,  '-o',  color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, snip_rigl_before_all_scores_10to1,  '--o',  color='#228B22',linewidth=linewidth, markersize=markersize )
# roberta_large.plot(x_axis, omp_after_all_scores_10to1,  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, random_before_all_scores_10to1,  '-o',  color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, random_all_after_scores_10to1,  '-o',  color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, random_rigl_all_cores_10to1,  '--o',   color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, gmp_all_scores_10to1,  '-o',   color='#CD00CD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, omp_before_all_scores_10to1,  '-o',  color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, omp_rigl_all_before_scores_10to1,  '--o',  color='#bcbd22',linewidth=linewidth, markersize=markersize )
#
#
# roberta_large.set_title('Trained on 10-to-10 Tested on 10-to-10',fontsize=Titlesize)
# roberta_large.axes.get_xaxis().set_visible(True)
# roberta_large.set_ylabel(r'BLEU $\uparrow$', fontsize=Titlesize)
# roberta_large.xaxis.set_ticks(x_axis)
# roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
# roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# # xposition = [3,7]
# # for xc in xposition:
# #     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
# roberta_large.grid(True, linestyle='-', linewidth=0.5, )
#
# roberta_large.spines['right'].set_visible(False)
# roberta_large.spines['top'].set_visible(False)



plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=6, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.05, bottom=0.29, right=0.95, top=0.94, wspace=0.2, hspace=0.35)
plt.savefig('multilingual_test_all.pdf')
plt.show()


