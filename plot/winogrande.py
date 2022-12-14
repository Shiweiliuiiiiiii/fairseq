import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 4
linewidth = 1.5
dense = [77.31]

acc_gm = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_gm_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_gm.append(float(line.split()[19][:-1]))


acc_omg_after = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_gm_after.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            print(line.split())
            acc_omg_after.append(float(line.split()[19][:-1]))


acc_snip = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_snip_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_snip.append(float(line.split()[19][:-1]))

acc_snip_rigl = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_snip_rigl.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_snip_rigl.append(float(line.split()[19][:-1]))


acc_omp_rigl = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_gm_rigl.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_omp_rigl.append(float(line.split()[19][:-1]))


acc_imp = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_imp_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_imp.append(float(line.split()[19][:-1]))

acc_random = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/cobert_winogrande_random.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_random.append(float(line.split()[19][:-1]))

acc_random_after = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_random_after.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            # print(line.split())
            acc_random_after.append(float(line.split()[19][:-1]))

acc_random_rigl = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_random_rigl.out') as file:
    for line in file:
        if 'val | epoch 001 |' in line:
            # print(line.split())
            acc_random_rigl.append(float(line.split()[19][:-1]))

acc_gmp = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/winogrande/winogrande_gmp.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            # print(line.split())
            acc_gmp.append(float(line.split()[19][:-1]))





x_axis = range(10)


# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, [acc_imp[0]]*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_imp[1:],  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_omg_after,  '-o',   label='One-Shot LRR (After)',color='#77AC30',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_random_after,  '-o',   label='Random LRR (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_gmp,  '-o',   label='GMP (During)',color='magenta',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, acc_snip,  '-o',   label='SNIP (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_snip_rigl,  '--o',   label='SNIP+RIGL (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, acc_random,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_random_rigl,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, acc_gm,  '-o',   label='OMP (Before)' ,color='cyan',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_omp_rigl,  '--o',   label='OMP+RIGL (Before)',color='cyan',linewidth=linewidth, markersize=markersize )


# roberta_large.plot(x_axis, [acc_imp[0]]*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
# vgg_all.plot(x_axis, [50]*11,  '-o',   label='random guess',color='gray',linewidth=3, markersize=markersize, )
roberta_large.set_title('Roberta Large on WinoGrande',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('Roberta_WinoGrande.pdf')
plt.show()