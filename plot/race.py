import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 3
linewidth =2
dense = [77.31]

acc_test = []
acc_test1 = []
with open('/Users/liushiwei/Projects/TUE_projects/fairseq/results/cobert_race_TEST.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test1.append(float(line.split()[19]))

acc_test_random = []
acc_test_random_high = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_random_test.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_random.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_random_high.append(float(line.split()[19]))

acc_test_gmp = []
acc_test_gmp_high = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_gmp.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_gmp.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_gmp_high.append(float(line.split()[19]))


acc_test_random_omp_after = []
acc_test_random_omp_after_high = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_omp_random_after_ck3.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_random_omp_after.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_random_omp_after_high.append(float(line.split()[19]))


acc_test_random_omp_rigl = []
acc_test_random_omp_rigl_high = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_omp_random_rigl_after_ck3.out') as file:
    for line in file:
        if 'test | valid on' in line:
            acc_test_random_omp_rigl.append(float(line.split()[19]))
        if ' | test1 | valid on' in line:
            acc_test_random_omp_rigl_high.append(float(line.split()[19]))



acc_test_imp = acc_test[:10]
acc_test_imp.append(26.1)
acc_test_omp = acc_test[10:20]
acc_test_snip = acc_test[20:29]
acc_test_snip.append(26.0)

acc_test_imp_high = [82, 82.6, 81.4, 79.1, 76, 71.6, 63.9, 56.3, 51.6, 36.8 ]
acc_test_imp_high.append(26.1)
acc_test_omp_high = acc_test1[10:20]
acc_test_snip_high = acc_test1[20:29]
acc_test_snip_high.append(25.0)

# after
# acc_test_random_after = acc_test_random_omp_after[:10]
# acc_test_omp_after = acc_test_random_omp_after[10:]

# acc_test_random_after_high = acc_test_random_omp_after_high[:10]
# acc_test_omp_after_high = acc_test_random_omp_after_high[10:]

# rigl
acc_test_random_rigl = acc_test_random_omp_rigl[:10]
acc_test_omp_rigl = acc_test_random_omp_rigl[10:]

acc_test_random_rigl_high = acc_test_random_omp_rigl_high[:10]
acc_test_omp_rigl_high = acc_test_random_omp_rigl_high[10:]

robert_snip_rigl = []

x_axis = range(10)


# high
# roberta_large = fig.add_subplot(1,1,1)
# roberta_large.plot(x_axis, [acc_test_imp_high[0]]*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_imp_high[1:],  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, acc_test_omp_after_high,  '-o',   label='One-Shot LRR (After)',color='#77AC30',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, acc_test_random_after_high,  '-o',   label='Random LRR (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_gmp_high,  '-o',   label='GMP (During)',color='magenta',linewidth=linewidth, markersize=markersize, )
#
#
# roberta_large.plot(x_axis, acc_test_snip_high,  '-o',   label='SNIP (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, robert_snip_rigl,  '--o',   label='SNIP+RIGL (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize )
# roberta_large.plot(x_axis, acc_test_random_high,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_random_rigl_high,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, acc_test_omp_high,  '-o',   label='OMP (Before)' ,color='cyan',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_omp_rigl_high,  '--o',   label='OMP+RIGL (Before)',color='cyan',linewidth=linewidth, markersize=markersize )


# middle
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, [acc_test_imp[0]]*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_imp[1:],  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_omp_after_high,  '-o',   label='One-Shot LRR (After)',color='#77AC30',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_random_after_high,  '-o',   label='Random LRR (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_gmp,  '-o',   label='GMP (During)',color='magenta',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, acc_test_snip,  '-o',   label='SNIP (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_snip_rigl,  '--o',   label='SNIP+RIGL (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, acc_test_random,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_random_rigl,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, acc_test_omp,  '-o',   label='OMP (Before)' ,color='cyan',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_omp_rigl,  '--o',   label='OMP+RIGL (Before)',color='cyan',linewidth=linewidth, markersize=markersize )




roberta_large.set_title('Roberta Large on RACE (Middle)',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('Roberta_race_middle.pdf')
plt.show()