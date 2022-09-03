import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 3

dense = [77.31]

robert_lth = [75.92, 74.04, 69.62, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
x_lth = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]

robert_snip = [20.31, 18.92, 18.51, 19.16, 18.67, 18.76, 18.76, 18.59, 18.59, 18.26]
robert_gm_before = [75.92, 59.41, 39.39, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_after = [62.24, 29.07, 20.63, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_rigl = [60.36, 29.57, 21.38, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_random_before = [28.99, 20.23, 21.87, 21.29, 20.97, 20.07, 20.48, 20.31, 20.97, 20.72]
robert_random_after  =  [18.10, 19.73, 19.25, 18.76, 20.23, 20.07, 20.48, 20.31, 19.00, 20.07]
# 0.02 0.05 0.1 ... 0.9 0.95
x_axis = range(10)


# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, dense*10,  '-o',   label='Dense model',color='black',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip,  '-',   label='SNIP',color='#dbb40c',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth,  '-',   label='LTH',color='orange',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl,  '-',   label='OMP+RIGL',color='red',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before,  '-',   label='OMP Before',color='green',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_after,  '-',   label='OMP After',color='cyan',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before,  '-',   label='Random Before',color='purple',linewidth=3, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after,  '-',   label='Random After',color='yellow',linewidth=3, markersize=markersize, )
# vgg_all.plot(x_axis, [50]*11,  '-o',   label='random guess',color='gray',linewidth=3, markersize=markersize, )
roberta_large.set_title('Roberta large on CommonsenseQA',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('Roberta_CommonsenseQA.png')
plt.show()