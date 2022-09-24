import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 14
Titlesize = 18
markersize = 7
linewidth = 2.2


# commonsenseQA
dense_csqa = [77.31]
x_lth_csqa = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]


robert_lth_csqa = [76.57, 74.11, 71.66, 66.17, 51.35, 40.62, 19.57, 19.57, 19.57, 19.57]
robert_gm_after_csqa  = [76.49, 74.61, 71.33, 54.13, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_snip_csqa = [73.87, 66.74, 20.96, 20.80, 20.06, 19.90, 26.12, 19.57, 19.57, 19.57 ]
robert_gmp_csqa  = [76.24, 75.10, 69.86, 59.70, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_before_csqa  = [75.51, 51.43, 27.60, 28.09, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]





x_axis = range(10)




roberta_large = fig.add_subplot(1,2,1)
roberta_large.plot(x_axis, dense_csqa*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(x_axis, robert_snip_csqa,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_csqa,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_after_csqa,  '-o',   label='One-Shot LRR (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp_csqa,  '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_csqa,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )


roberta_large.set_title('Roberta on CommonsenseQA',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)

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




# roberta_large = fig.add_subplot(1,2,2)
# roberta_large.plot(x_axis, 100*np.array([acc_imp_wino[0]]*10),  '-o', color='black',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, 100*np.array(acc_imp_wino[1:]),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, 100*np.array(acc_omg_after_wino),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, 100*np.array(acc_random_after_wino),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, 100*np.array(acc_gmp_wino),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
#
#
# roberta_large.plot(x_axis, 100*np.array(acc_snip_wino),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, 100*np.array(acc_snip_rigl_wino),  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
# roberta_large.plot(x_axis, 100*np.array(acc_random_wino),  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, 100*np.array(acc_random_rigl_wino),  '--o', color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, 100*np.array(acc_gm_wino),  '-o',  color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, 100*np.array(acc_omp_rigl_wino),  '--o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize )
#
#
#
# roberta_large.set_title('Roberta on WinoGrande',fontsize=Titlesize)
# roberta_large.xaxis.set_ticks(x_axis, rotation=45)
# roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
# roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# roberta_large.axes.get_xaxis().set_visible(True)
# roberta_large.grid(True, linestyle='-', linewidth=0.5, )
# roberta_large.spines['right'].set_visible(False)
# roberta_large.spines['top'].set_visible(False)
#
# print((acc_imp_wino[0] - acc_imp_wino[4])/  acc_imp_wino[0])



plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=6, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95, wspace=0.2, hspace=0.35)
# roberta_large.set_title('Roberta large on CommonsenseQA',fontsize=fontsize)
# roberta_large.set_xticks(range(10))
# roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
#
# roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.savefig('CSQA_noembed.pdf')
plt.show()