import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 5.5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 12
Titlesize = 18
markersize = 7
linewidth = 2.2

# commonsenseQA
dense_csqa = [77.31]
x_lth_csqa = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]


robert_lth_csqa_noe = [76.57, 74.11, 71.66, 66.17, 51.35, 40.62, 19.57, 19.57, 19.57, 19.57]
robert_gm_after_csqa_noe  = [76.49, 74.61, 71.33, 54.13, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_snip_csqa_noe = [73.87, 66.74, 20.96, 20.80, 20.06, 19.90, 19.57, 19.57, 19.57, 19.57 ]
robert_gmp_csqa_noe  = [76.24, 75.10, 69.86, 59.70, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_before_csqa_noe  = [75.51, 51.43, 27.60, 28.09, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]

robert_lth_csqa = [75.92, 74.04, 69.62, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_snip_csqa = [20.31, 18.92, 18.51, 19.16, 18.67, 18.76, 18.76, 18.59, 18.59, 18.26]
robert_gmp_csqa  = [76.00, 72.40, 63.96, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_before_csqa = [75.92, 59.41, 39.39, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gm_after_csqa  = [76.99, 74.86, 55.61, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]


x_axis = range(10)


roberta_large = fig.add_subplot(1,2,1)
roberta_large.plot(x_axis, dense_csqa*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_csqa,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_csqa_noe,  '--*',   label='SNIP w/o E (Before) ',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_csqa,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_csqa_noe,  '--*',   label='LTH w/o E (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_after_csqa,  '-o',   label='OMP (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_after_csqa_noe,  '--*',   label='OMP w/o E (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp_csqa,  '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp_csqa_noe,  '--*',   label='GMP w/o E (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_csqa,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_csqa_noe,  '--*',   label='OMP w/o E (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )


roberta_large.set_title('RoBERTa on CommonsenseQA',fontsize=Titlesize)
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



dense_g2t = [43.4]
x_lth_csqa = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]


robert_lth_g2t_noe = [43.1, 42.0, 42.3, 40.2, 39.1, 37.6, 33.8, 30.3, 29.4, 27.6 ]
robert_snip_g2t_noe = [42.2, 39.7, 34.3, 34.5, 32.8, 33.3, 33.9, 34.1, 36.0, 34.5]
robert_gmp_g2t_noe  = [41.2, 42.4, 41.7, 39.5, 41.1, 39.1, 35.8, 34.5, 35.1, 33.8]
robert_gm_before_g2t_noe  = [42.7, 40.3, 33.3, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]
robert_gm_after_g2t_noe  = [43.1, 42.2, 43.5, 42.0, 35.1, 34.6, 31.3, 36.0, 33.6, 35.4]


robert_gmp_svamp =           [42.0, 41.1, 41.7, 40.5, 39.9, 38.8, 35.7, 33.1, 34.8, 33.6]
robert_snip_svamp =          [33.9, 33.8, 31.0, 33.7, 33.8, 33.2, 33.9, 34.7, 33.7, 35.0]
robert_lth_svamp =           [43.3, 41.1, 40.6, 39.9, 38.7, 33.1, 32.3, 29.3, 29.0, 27.1]
robert_gm_after_svamp =      [44.4, 43.1, 42.7, 40.7, 35.3, 33.1, 35.1, 34.3, 34.1, 33.0]
robert_gm_before_svamp =     [38.5, 29.1, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]




roberta_large = fig.add_subplot(1,2,2)
roberta_large.plot(x_axis, np.array(dense_g2t*10),  '-o', color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(robert_lth_g2t_noe),  '--*', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(robert_gm_after_g2t_noe),  '--*', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(robert_gmp_g2t_noe),  '--*', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(robert_snip_g2t_noe),  '--*', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_g2t_noe,  '--*' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, np.array(robert_lth_svamp),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(robert_gm_after_svamp),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(robert_gmp_svamp),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(robert_snip_svamp),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_svamp, '-o', color='#bcbd22',linewidth=linewidth, markersize=markersize, )



roberta_large.set_title('GTS on SVAMP',fontsize=Titlesize)
roberta_large.xaxis.set_ticks(x_axis, rotation=45)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=5, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.15, bottom=0.27, right=0.85, top=0.94, wspace=0.3, hspace=0.35)
# roberta_large.set_title('Roberta large on CommonsenseQA',fontsize=fontsize)
# roberta_large.set_xticks(range(10))
# roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
#
# roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.savefig('CSQA_noembed.pdf')
plt.show()