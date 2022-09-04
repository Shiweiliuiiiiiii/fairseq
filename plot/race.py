import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 3

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



acc_test_imp = acc_test[:10]
acc_test_imp.append(26.1)
acc_test_gm = acc_test[10:20]
acc_test_snip = acc_test[20:29]
acc_test_snip.append(26.0)

acc_test_imp_high = [82, 82.6, 81.4, 79.1, 76, 71.6, 63.9, 56.3, 51.6, 36.8 ]
acc_test_imp_high.append(26.1)
acc_test_gm_high = acc_test1[10:20]
acc_test_snip_high = acc_test1[20:29]
acc_test_snip_high.append(25.0)

x_axis = range(10)


# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, acc_test_snip_high,  '-',   label='SNIP',color='#dbb40c',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_imp_high[1:],  '-',   label='LTH',color='orange',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_gm_high,  '-',   label='OMP Before',color='green',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, acc_test_random_high,  '-',   label='Random Before',color='purple',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, [acc_test_imp_high[0]]*10,  '-o',   label='Dense model',color='black',linewidth=3, markersize=markersize, )
# vgg_all.plot(x_axis, [50]*11,  '-o',   label='random guess',color='gray',linewidth=3, markersize=markersize, )
roberta_large.set_title('Roberta Large on RACE high',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('Roberta_race_high.png')
plt.show()