import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 3

dense = [77.31]

acc_gm = []
with open('/home/shiweiliu/PycharmProjects/fairseq/winogrande_gm_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_gm.append(float(line.split()[19][:-1]))

acc_snip = []
with open('/home/shiweiliu/PycharmProjects/fairseq/winogrande_snip_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_snip.append(float(line.split()[19][:-1]))

acc_imp = []
with open('/home/shiweiliu/PycharmProjects/fairseq/winogrande_imp_0.2.out') as file:
    for line in file:
        if 'val | epoch 019 |' in line:
            acc_imp.append(float(line.split()[19][:-1]))

x_axis = range(10)


# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, acc_snip,  '-',   label='SNIP',color='#dbb40c',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, acc_imp[1:],  '-',   label='LTH',color='orange',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, acc_gm,  '-',   label='Global Magnitude',color='green',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, [acc_imp[0]]*10,  '-o',   label='Dense model',color='black',linewidth=3, markersize=markersize, )
# vgg_all.plot(x_axis, [50]*11,  '-o',   label='random guess',color='gray',linewidth=3, markersize=markersize, )
roberta_large.set_title('Roberta Large on WinoGrande',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('Roberta_WinoGrande.png')
plt.show()