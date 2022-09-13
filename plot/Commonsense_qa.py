import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 15
markersize = 8
linewidth = 2
dense = [77.31]

robert_lth = [75.92, 74.04, 69.62, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
x_lth = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]

robert_snip = [20.31, 18.92, 18.51, 19.16, 18.67, 18.76, 18.76, 18.59, 18.59, 18.26]
robert_gmp = [76.00, 72.40, 63.96, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_snip_rigl = [19.00, 19.00, 20.22, 17.60, 22.19, 19.73, 18.75, 19.25, 20.14, 19.57]
robert_gm_before = [75.92, 59.41, 39.39, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_after = [62.24, 29.07, 20.63, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_rigl = [60.36, 29.57, 21.38, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_random_before = [28.99, 20.23, 21.87, 21.29, 20.97, 20.07, 20.48, 20.31, 20.97, 20.72]
robert_random_rigl = [24.82, 19.57, 21.54, 20.31, 20.23, 22.35, 20.39, 20.72, 20.31, 20.72]
robert_random_after  =  [18.10, 19.73, 19.25, 18.76, 20.23, 20.07, 20.48, 20.31, 19.00, 20.07]
# 0.02 0.05 0.1 ... 0.9 0.95
x_axis = range(10)


# prune
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, dense*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip,  '-',   label='SNIP',color='blue',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp,  '-',   label='GMP',color='yellow',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl,  '-',   label='SNIP+RIGL',color='blue',linewidth=linewidth, markersize=markersize, marker='^'  )
roberta_large.plot(x_axis, robert_lth,  '-',   label='LTH',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before,  '-',   label='OMP Before',color='green',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_after,  '--',   label='OMP After',color='green',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl,  '-',   label='OMP+RIGL',color='green',linewidth=linewidth, markersize=markersize, marker='^' )
roberta_large.plot(x_axis, robert_random_before,  '-',   label='Random Before',color='purple',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after,  '--',   label='Random After',color='purple',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl,  '-',   label='Random+RIGL',color='purple',linewidth=linewidth, markersize=markersize, marker='^' )


roberta_large.set_title('Roberta large on CommonsenseQA',fontsize=fontsize)
roberta_large.set_xticks(range(10))
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.legend()
plt.savefig('Roberta_CommonsenseQA.png')
plt.show()