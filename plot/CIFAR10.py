import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 12
Titlesize = 18
markersize = 7
linewidth = 2.2
x_axis = range(10)


# commonsenseQA
dense_csqa = [66.91]
x_lth_csqa = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]


robert_gmp = [64.78, 56.76, 43.49, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_obert  = [66.50, 65.11, 62.00, 52.91, 37.10, 25.88, 22.19, 19.90, 16.54, 20.07]
robert_lth = [66.91, 64.94, 59.37, 32.84, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]


roberta_large = fig.add_subplot(1,2,1)
roberta_large.plot(x_axis, dense_csqa*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_obert,  '-o',   label='oBERT',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp,  '-o',   label='GMP',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth,  '-o',   label='LTH',color='purple',linewidth=linewidth, markersize=markersize, )


roberta_large.set_title('CommonsenseQA',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity', fontsize=Titlesize)

# print((dense_csqa[0] - robert_lth_csqa[3])/  dense_csqa)[]
roberta_large.xaxis.set_ticks(x_axis)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )

roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



dense_race = [68.09]
x_lth_race = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]
robert_race_gmp = [67.02, 65.27, 62.14, 56.78, 52.82, 50.33, 50.61, 51.07, 51.98, 49.64]
robert_race_obert  = [66.64, 65.72, 61.38, 57.24, 52.51, 51.06, 50.33, 50.55, 50.46, 50.45]
robert_race_lth = [67.17, 65.72, 61.99, 56.55, 54.75, 51.93, 51.77, 51.65, 51.00, 50.22 ]


roberta_large = fig.add_subplot(1,2,2)
roberta_large.plot(x_axis, dense_race*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_race_obert,  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_race_gmp,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_race_lth,  '-o',  color='purple',linewidth=linewidth, markersize=markersize, )


roberta_large.set_title('Winogrande',fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity', fontsize=Titlesize)
roberta_large.xaxis.set_ticks(x_axis, rotation=45)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=6, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.05, bottom=0.29, right=0.95, top=0.94, wspace=0.2, hspace=0.35)
plt.savefig('Roberta_base_oBERT.pdf', bbox_inches='tight')
plt.show()
