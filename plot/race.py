import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 3

dense = [77.31]

# acc = []
# with open('/home/shiweiliu/PycharmProjects/fairseq/cobert_race_TEST.out') as file:
#     for line in file:
#         if '| INFO | test |' in line:
#             acc.append(float(line.split()[19][:-1]))
acc_imp = [86.6, 87.7, 88, 86.4, 83.4, 77.6, 71.4, 64.1, 57.7, 39.1, 36.8 ]
acc_gm_middle = [25.4, 25.8, 26, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5  ]
acc_snip_middle = [27.7, 25, 24.3, 24.9, 26.4, 27.1 , 25.1 , 22.4, 25.6, 25.3,   ]



x_axis = range(10)


# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, acc_snip_middle,  '-',   label='SNIP',color='#dbb40c',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, acc_imp[1:],  '-',   label='LTH',color='orange',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, acc_gm_middle,  '-',   label='Global Magnitude',color='green',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, [acc_imp[0]]*10,  '-o',   label='Dense model',color='black',linewidth=3, markersize=markersize, )
# vgg_all.plot(x_axis, [50]*11,  '-o',   label='random guess',color='gray',linewidth=3, markersize=markersize, )
roberta_large.set_title('Roberta Large on RACE',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('Roberta_race.png')
plt.show()