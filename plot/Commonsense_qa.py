import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 10), dpi=150, facecolor='w', edgecolor='k')
fontsize = 14
Titlesize = 18
markersize = 7
linewidth = 2.2


# commonsenseQA
dense_csqa = [77.31]

robert_lth_csqa = [75.92, 74.04, 69.62, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
x_lth_csqa = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]

robert_snip_csqa = [20.31, 18.92, 18.51, 19.16, 18.67, 18.76, 18.76, 18.59, 18.59, 18.26]
robert_gmp_csqa  = [76.00, 72.40, 63.96, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_snip_rigl_csqa = [19.00, 19.00, 20.22, 17.60, 22.19, 19.73, 18.75, 19.25, 20.14, 19.57]
robert_gm_before_csqa = [75.92, 59.41, 39.39, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_after_csqa  = [76.99, 74.86, 55.61, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_rigl_csqa   = [60.36, 29.57, 21.38, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_random_before_csqa = [28.99, 20.23, 21.87, 21.29, 20.97, 20.07, 20.48, 20.31, 20.97, 20.72]
robert_random_rigl_csqa   = [24.82, 19.57, 21.54, 20.31, 20.23, 22.35, 20.39, 20.72, 20.31, 20.72]
robert_random_after_csqa  = [36.61, 25.06, 21.95, 20.39, 20.88, 20.31, 19.74, 20.80, 21.29, 21.05]



# winogrande

acc_gm_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_gm_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_gm_wino.append(float(line.split()[19][:-1]))


acc_omg_after_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_gm_after.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            print(line.split())
            acc_omg_after_wino.append(float(line.split()[19][:-1]))


acc_snip_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_snip_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_snip_wino.append(float(line.split()[19][:-1]))

acc_snip_rigl_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_snip_rigl.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_snip_rigl_wino.append(float(line.split()[19][:-1]))


acc_omp_rigl_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_gm_rigl.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_omp_rigl_wino.append(float(line.split()[19][:-1]))


acc_imp_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_imp_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_imp_wino.append(float(line.split()[19][:-1]))

acc_random_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/cobert_winogrande_random.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_random_wino.append(float(line.split()[19][:-1]))

acc_random_after_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_random_after.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            # print(line.split())
            acc_random_after_wino.append(float(line.split()[19][:-1]))

acc_random_rigl_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_random_rigl.out') as file:
    for line in file:
        if 'val | epoch 001 |' in line:
            # print(line.split())
            acc_random_rigl_wino.append(float(line.split()[19][:-1]))

acc_gmp_wino = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_gmp.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            # print(line.split())
            acc_gmp_wino.append(float(line.split()[19][:-1]))



# RACE

acc_test_RACE = []
acc_test1_RACE = []
with open('/Users/liushiwei/Projects/TUE_projects/fairseq/results/cobert_race_TEST.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_RACE.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test1_RACE.append(float(line.split()[19]))

acc_test_random_RACE = []
acc_test_random_high_RACE = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_random_test.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_random_RACE.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_random_high_RACE.append(float(line.split()[19]))

acc_test_gmp_RACE = []
acc_test_gmp_high_RACE = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_gmp.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_gmp_RACE.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_gmp_high_RACE.append(float(line.split()[19]))


acc_test_random_omp_after_RACE = []
acc_test_random_omp_after_high_RACE = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_omp_random_after_ck3.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_random_omp_after_RACE.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_random_omp_after_high_RACE.append(float(line.split()[19]))


acc_test_random_omp_rigl_RACE = []
acc_test_random_omp_rigl_high_RACE = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_omp_random_rigl_after_ck3.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_random_omp_rigl_RACE.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_random_omp_rigl_high_RACE.append(float(line.split()[19]))

acc_test_snip_rigl_RACE = []
acc_test_snip_rigl_high_RACE = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_snip_rigl_test.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_snip_rigl_RACE.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_snip_rigl_high_RACE.append(float(line.split()[19]))

acc_test_snip_rigl_RACE = []
acc_test_snip_rigl_high_RACE = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_snip_rigl_test.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_snip_rigl_RACE.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_snip_rigl_high_RACE.append(float(line.split()[19]))

acc_test_omp_random_after_RACE = []
acc_test_omp_random_after_high_RACE = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_omp_random_after_test.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_omp_random_after_RACE.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_omp_random_after_high_RACE.append(float(line.split()[19]))




acc_test_imp_RACE = acc_test_RACE[:10]
acc_test_imp_RACE.append(26.1)
acc_test_omp_RACE = acc_test_RACE[10:20]
acc_test_snip_RACE = acc_test_RACE[20:29]
acc_test_snip_RACE.append(26.0)

acc_test_imp_high_RACE = [82, 82.6, 81.4, 79.1, 76, 71.6, 63.9, 56.3, 51.6, 36.8 ]
acc_test_imp_high_RACE.append(26.1)
acc_test_omp_high_RACE = acc_test1_RACE[10:20]
acc_test_snip_high_RACE = acc_test1_RACE[20:29]
acc_test_snip_high_RACE.append(25.0)

# after
acc_test_random_after_RACE = acc_test_omp_random_after_RACE[10:]
acc_test_omp_after_RACE =  acc_test_omp_random_after_RACE[:10]

acc_test_random_after_high_RACE = acc_test_omp_random_after_high_RACE[10:]
acc_test_omp_after_high_RACE = acc_test_omp_random_after_high_RACE[:10]

# rigl
acc_test_random_rigl_RACE = acc_test_random_omp_rigl_RACE[:10]
acc_test_omp_rigl_RACE = acc_test_random_omp_rigl_RACE[10:]

acc_test_random_rigl_high_RACE = acc_test_random_omp_rigl_high_RACE[:10]
acc_test_omp_rigl_high_RACE = acc_test_random_omp_rigl_high_RACE[10:]

x_axis = range(10)




roberta_large = fig.add_subplot(2,2,1)
roberta_large.plot(x_axis, dense_csqa*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, robert_snip_csqa,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_csqa,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_rigl_csqa,  '--o',   label='SNIP+RIGL (Before)',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_csqa,  '-o',   label='One-Shot LRR (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, robert_random_before_csqa,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_csqa,  '-o',   label='Random LRR (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_random_rigl_csqa,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_csqa,  '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_gm_before_csqa,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl_csqa,  '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('Roberta on CommonsenseQA',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(False)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
# roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
# roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

# vgg_all.set_yticks([16,10,0,-10,-20,-30,-40])
# vgg_all.set_yticklabels(['',10,0,-10,-20,-30,-40],fontsize=fontsize )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)




roberta_large = fig.add_subplot(2,2,2)
roberta_large.plot(x_axis, 100*np.array([acc_imp_wino[0]]*10),  '-o', color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, 100*np.array(acc_imp_wino[1:]),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, 100*np.array(acc_omg_after_wino),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, 100*np.array(acc_random_after_wino),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, 100*np.array(acc_gmp_wino),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, 100*np.array(acc_snip_wino),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, 100*np.array(acc_snip_rigl_wino),  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, 100*np.array(acc_random_wino),  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, 100*np.array(acc_random_rigl_wino),  '--o', color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, 100*np.array(acc_gm_wino),  '-o',  color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, 100*np.array(acc_omp_rigl_wino),  '--o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('Roberta on WinoGrande',fontsize=Titlesize)
# roberta_large.set_xticks(range(10))
# roberta_large.set_ylabel('Accuracy [%]', fontsize=fontsize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
# roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
# roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

# vgg_all.set_yticks([16,10,0,-10,-20,-30,-40])
# vgg_all.set_yticklabels(['',10,0,-10,-20,-30,-40],fontsize=fontsize )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
roberta_large.axes.get_xaxis().set_visible(False)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


roberta_large = fig.add_subplot(2,2,3)

roberta_large.plot(x_axis, [acc_test_imp_RACE[0]]*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_imp_RACE[1:],  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_omp_after_RACE,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_random_after_RACE,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_gmp_RACE,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, acc_test_snip_RACE,  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_snip_rigl_RACE,  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, acc_test_random_RACE,  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_random_rigl_RACE, '--o', color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, acc_test_omp_RACE,  '-o', color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_omp_rigl_RACE,  '--o', color='#bcbd22',linewidth=linewidth, markersize=markersize )



roberta_large.set_title('Roberta on RACE (Middle)',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

# vgg_all.set_yticks([16,10,0,-10,-20,-30,-40])
# vgg_all.set_yticklabels(['',10,0,-10,-20,-30,-40],fontsize=fontsize )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


roberta_large = fig.add_subplot(2,2,4)

roberta_large.plot(x_axis, [acc_test_imp_high_RACE[0]]*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_imp_high_RACE[1:],  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_random_after_high_RACE,  '-o',   color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_omp_after_high_RACE,  '-o',   color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_gmp_high_RACE,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, acc_test_snip_high_RACE,  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_snip_rigl_high_RACE,  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, acc_test_random_high_RACE,  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_random_rigl_high_RACE,  '--o', color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, acc_test_omp_high_RACE,  '-o', color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_omp_rigl_high_RACE,  '--o', color='#bcbd22',linewidth=linewidth, markersize=markersize )



roberta_large.set_title('Roberta on RACE (High)',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

# vgg_all.set_yticks([16,10,0,-10,-20,-30,-40])
# vgg_all.set_yticklabels(['',10,0,-10,-20,-30,-40],fontsize=fontsize )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=6, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
# roberta_large.set_title('Roberta large on CommonsenseQA',fontsize=fontsize)
# roberta_large.set_xticks(range(10))
# roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
#
# roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.savefig('Roberta_commonsense_reasoning.pdf')
plt.show()