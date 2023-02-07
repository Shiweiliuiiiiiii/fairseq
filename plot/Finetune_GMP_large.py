import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 12
Titlesize = 10
markersize = 7
linewidth = 2.2
x_axis = range(10)

## LLR refers to LTH + learning rate rewinding

# commonsenseQA
dense_csqa = [77.31]
x_lth_csqa = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]

robert_gmp_csqa_5epochs  = [76.00, 72.40, 63.96, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gmp_csqa_no_embed_5epochs  = [76.24, 75.10, 69.86, 59.70, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]
robert_gmp_csqa_no_embed_no_classifier_5epochs = [76.00, 75.10, 71.74, 62.08, 41.68, 28.17, 19.73, 19.57, 19.81, 18.50]
robert_gmp_LLR_no_embed_no_classifier_5epochs  = [76.49, 74.12, 70.51, 64.78, 54.95, 43.57, 36.77, 30.63, 22.11, 19.57]
robert_obert_LLR_no_embed_no_classifier_5epochs_noKD  = [72, 73.2, 70.2, 70, 67.7, 58.7, 54.2, 48.3, 47.4, 45.5]
robert_obert_LLR_no_embed_no_classifier_5epochs  = [74, 74.2, 72.2, 71, 68.7, 60.7, 56.2, 52.3, 51.4, 48]
robert_obert_plain_5epochs  = [71, 70.7, 70.8, 69.1, 63.1 ,51.2, 32.6, 30.8, 22.5, 22.2]

# robert_lth_csqa = [75.92, 74.04, 69.62, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57, 19.57]

robert_gmp_104_300_no_embed_no_classifier_5epochs = [20.06, 20.39, 21.13, 21.37, 20.55, 21.78, 21.94, 20.72, 21.37, 16.95]
robert_gmp_104_600_no_embed_no_classifier_5epochs = [21.21, 20.96, 21.37, 18.34, 19.73, 20.63, 19.90, 21.37, 18.75, 18.34]
robert_gmp_505_300_no_embed_no_classifier_5epochs = [19.32, 17.19, 20.88, 20.06, 23.34, 21.13, 19.81, 20.22, 19.73, 19.00]
robert_gmp_505_600_no_embed_no_classifier_5epochs = [20.63, 20.31, 17.69, 20.47, 20.39, 19.73, 19.65, 20.06, 19.08, 21.04]
robert_gmp_105_300_no_embed_no_classifier_5epochs = [76.90, 75.51, 71.82, 65.52, 55.61, 39.72, 32.18, 21.13, 20.63, 19.98]
robert_gmp_105_600_no_embed_no_classifier_5epochs = [76.00, 75.10, 71.74, 62.08, 41.68, 28.17, 19.73, 19.57, 19.81, 18.50]
robert_gmp_506_300_no_embed_no_classifier_5epochs = [76.08, 73.95, 69.61, 57.90, 36.69, 23.51, 19.24, 21.86, 19.16, 19.49]
robert_gmp_506_600_no_embed_no_classifier_5epochs = [76.90, 72.40, 66.58, 27.68, 22.52, 24.73, 20.63, 20.47, 18.09, 18.91]


robert_gmp_104_300_no_embed_no_classifier_10epochs = [20.80, 20.55, 19.49, 21.62, 21.21, 19.57, 19.49, 19.16, 20.06, 19.98]
robert_gmp_104_600_no_embed_no_classifier_10epochs = [20.06, 20.22, 20.72, 19.73, 18.70, 21.04, 20.34, 19.00, 19.16, 20.55]
robert_gmp_505_300_no_embed_no_classifier_10epochs = [21.04, 20.39, 21.53, 21.37, 19.32, 20.06, 18.91, 19.65, 19.57, 18.34]
robert_gmp_505_600_no_embed_no_classifier_10epochs = [21.45, 19.90, 21.45, 19.08, 19.65, 19.57, 20.22, 19.49, 20.96, 21.13]
robert_gmp_105_300_no_embed_no_classifier_10epochs = [75.92, 73.95, 72.89, 69.69, 60.03, 50.69, 40.86, 34.39, 30.63, 22.03]
robert_gmp_105_600_no_embed_no_classifier_10epochs = [75.51, 74.03, 73.05, 69.53, 60.93, 50.20, 39.47, 31.53, 18.18, 19.98]
robert_gmp_506_300_no_embed_no_classifier_10epochs = [76.57, 74.49, 71.90, 65.02, 56.26, 41.67, 32.26, 18.83, 20.88, 23.75]
robert_gmp_506_600_no_embed_no_classifier_10epochs = [76.49, 74.61, 71.99, 64.53, 53.89, 38.49, 31.44, 22.85, 20.14, 20.63]



roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, dense_csqa*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp_csqa_5epochs,  '-o',   label='GMP',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp_csqa_no_embed_5epochs,  '-o',   label='GMP (not pruning emb)',color='blue',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gmp_csqa_no_embed_no_classifier_5epochs,  '-o',   label='GMP (not pruning emb and classifier)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_obert_plain_5epochs,  '--',   label='oBERT - KD - LRR (not pruning emb and classifier)',color='red',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gmp_105_600_no_embed_no_classifier_10epochs,  '-o',   label='GMP (no emb and no classifier) 10 epochs',color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gmp_LLR_no_embed_no_classifier_5epochs,  '-o',   label='GMP + LRR (not pruning emb and classifier)',color='purple',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_obert_LLR_no_embed_no_classifier_5epochs_noKD,  '--o',   label='oBERT - KD (not pruning emb and classifier)',color='red',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_obert_LLR_no_embed_no_classifier_5epochs,  '-o',   label='oBERT (not pruning emb and classifier)',color='red',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, robert_structured,  '-o',   label='Structrued (L1 Norm)',color='orange',linewidth=linewidth, markersize=markersize, )

# roberta_large.plot(x_axis, dense_csqa*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, robert_gmp_104_600_no_embed_no_classifier_5epochs,  '-o',   label='GMP lr=0.0001',color='#228B22',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, robert_gmp_505_600_no_embed_no_classifier_5epochs,  '-o',   label='GMP lr=0.00005',color='blue',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gmp_105_600_no_embed_no_classifier_5epochs,  '-o',   label='GMP (no emb and no classifier) 5epochs',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gmp_105_600_no_embed_no_classifier_10epochs,  '-o',   label='GMP (no emb and no classifier) 10epochs',color='orange',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, robert_gmp_105_600_no_embed_no_classifier_10epochs,  '-o',   label='GMP (no emb and no classifier) 10 epochs',color='orange',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, robert_gmp_506_600_no_embed_no_classifier_5epochs,  '-o',   label='GMP lr=0.000005',color='purple',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, robert_structured,  '-o',   label='Structrued (L1 Norm)',color='orange',linewidth=linewidth, markersize=markersize, )
#

roberta_large.set_title('CommonsenseQA (Roberta Large no emb and no classifier)',fontsize=Titlesize)
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


plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=4, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.05, bottom=0.29, right=0.95, top=0.94, wspace=0.2, hspace=0.35)
plt.savefig('GMP_10_epochs.pdf', bbox_inches='tight')
plt.show()
