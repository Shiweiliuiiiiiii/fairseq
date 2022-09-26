import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 10), dpi=150, facecolor='w', edgecolor='k')
fontsize = 12
Titlesize = 18
markersize = 7
linewidth = 2.2



# ESM-1F on HP-S2C5
dense_ESM_1F_HP_S2CS = [61.3]
x_axis = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]

LTH_ESM_1F_HP_S2CS = [62.50, 63.46, 59.62, 60.58, 61.54, 58.65, 60.58, 59.62, 59.62]
GMP_ESM_1F_HP_S2CS = []
OMP_after_ESM_1F_HP_S2CS   = [61.1, 60.8, 59.5, 61.0, 60.4, 60.7, 59.3, 59.1, 58.2]
OMP_before_ESM_1F_HP_S2CS   = []
OMP_before_rigl_ESM_1F_HP_S2CS   = []
RP_after_ESM_1F_HP_S2CS   = [60.6, 59.0, 58.3, 57.4, 57.7, 57.0, 58.0, 57.4]
RP_before_rigl_ESM_1F_HP_S2CS = [55.57, 52.93, 52.33, 53.43, 55.54, 55.37, 55.98]
SNIP_ESM_1F_HP_S2CS   = [60.3, 60.2, 59.0 , 59.2, 58.6, 59.3, 58.2, 58.0, 58.5]
SNIP_rigl_ESM_1F_HP_S2CS   = []

roberta_large = fig.add_subplot(2,2,1)
roberta_large.plot(range(len(x_axis)), dense_ESM_1F_HP_S2CS*10,  '-o',  color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(SNIP_ESM_1F_HP_S2CS)), SNIP_ESM_1F_HP_S2CS,  '-o',  color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(LTH_ESM_1F_HP_S2CS)), LTH_ESM_1F_HP_S2CS,   '-o',   color='orange',linewidth=linewidth, markersize=markersize, )

# roberta_large.plot(range(len(SNIP_rigl_HP_S)), SNIP_rigl_HP_S,   '--o',   label='SNIP+RIGL (Before)',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(range(len(OMP_after_ESM_1F_HP_S2CS)), OMP_after_ESM_1F_HP_S2CS,   '-o',   color='#00BFFF',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(range(len(RP_after_ESM_1F_HP_S2CS)), RP_after_ESM_1F_HP_S2CS,   '-o',  color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(range(len(SNIP_HP_S)), SNIP_HP_S,   '-o',   label='Random (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(RP_before_rigl_ESM_1F_HP_S2CS)), RP_before_rigl_ESM_1F_HP_S2CS,   '--o', color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(range(len(GMP_HP_S)), GMP_HP_S,   '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )

# roberta_large.plot(range(len(OMP_before_HP_S)), OMP_before_HP_S,   '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(range(len(OMP_before_rigl_HP_S)), OMP_before_rigl_HP_S,   '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('ESM-1F on HP-S$^2$C5',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
# roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.axes.get_yaxis().set_visible(True)
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



# gcn on HP-S2C5
dense_gcn_1F_HP_S2CS = []
x_axis = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]

LTH_gcn_1F_HP_S2CS = [74.04, 73.56, 72.12, 72.88, 71.83, 69.81, 68.40, 67.83, 66.04]
GMP_gcn_1F_HP_S2CS = [74.05, 66.63, 67.40, 67.79, 67.69, 67.12, 63.75, 65.00, 58.08]
OMP_after_gcn_1F_HP_S2CS   = [73.84, 71.82, 72.69, 72.30, 68.55, 66.25, 58.65, 40.76]
OMP_before_gcn_1F_HP_S2CS   = []
OMP_before_rigl_gcn_1F_HP_S2CS   = [73.65, 72.11, 71.34, 71.73, 68.65, 66.44, 61.05, 49.90]
RP_after_gcn_1F_HP_S2CS   = [68.08, 67.02, 66.92, 66.92, 65.10, 63.37, 63.27, 62.21]
RP_before_rigl_gcn_1F_HP_S2CS = [67.12, 66.73, 65.77, 65.58, 65.38, 63.85, 63.75, 63.17, 62.31]
SNIP_gcn_1F_HP_S2CS   = [70.38, 69.42, 70.77, 69.42, 66.25, 66.15, 65.09, 66.92, 65.48]
SNIP_rigl_gcn_1F_HP_S2CS   = [70.10, 70.38, 70.48, 69.90, 67.50, 57.21, 66.44, 65.29, 65.19]


roberta_large = fig.add_subplot(2,2,2)
# roberta_large.plot(range(len(x_axis)), dense_ESM_1F_HP_S2CS*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(SNIP_gcn_1F_HP_S2CS)), SNIP_gcn_1F_HP_S2CS,  '-o',  color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(LTH_gcn_1F_HP_S2CS)), LTH_gcn_1F_HP_S2CS,   '-o',  color='orange',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(SNIP_rigl_gcn_1F_HP_S2CS)), SNIP_rigl_gcn_1F_HP_S2CS,   '--o',  color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(range(len(OMP_after_gcn_1F_HP_S2CS)), OMP_after_gcn_1F_HP_S2CS,   '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(RP_after_gcn_1F_HP_S2CS)), RP_after_gcn_1F_HP_S2CS,   '-o',   color='#0072BD',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(RP_before_rigl_gcn_1F_HP_S2CS)), RP_before_rigl_gcn_1F_HP_S2CS,   '--o',  color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(range(len(GMP_gcn_1F_HP_S2CS)), GMP_gcn_1F_HP_S2CS,   '-o',   color='#CD00CD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(range(len(OMP_before_HP_S)), OMP_before_HP_S,   '-o',  color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(OMP_before_rigl_gcn_1F_HP_S2CS)), OMP_before_rigl_gcn_1F_HP_S2CS,   '--o',  color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GCN on HP-S$^2$C5',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
# roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.axes.get_yaxis().set_visible(True)
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



# ESM-1b on HP-S
dense_ESM1b_HP_S = [67.86]
x_axis = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]

LTH_HP_S = [68.48, 64.48, 53.33, 57.40]
GMP_HP_S = [67.7, 65.59, 57.11, 52.23, 50.52, 48.13, 45.14, 43.94]
OMP_after_HP_S   = [68.51, 62.92, 55.2, 51.83, 47.70, 47.73, 45.03, 45.06, 45.03]
OMP_before_HP_S   = [66.84, 60.82, 55.35, 43.90, 45.54, 47.95, 45.08, 45.23]
OMP_before_rigl_HP_S   = [65.00, 56.58, 47.81, 50.37, 49.15, 47.72, 48.15, 46.78]
RP_after_HP_S   = [65.21, 56.83, 54.65, 53.54, 54.20, 54.52, 54.59]
RP_before_rigl_HP_S = [55.57, 52.93, 52.33, 53.43, 55.54, 55.37, 55.98]
SNIP_HP_S   = [59.37, 58.41, 58.04, 57.4, 56.68, 56.16, 55.01, 54.27]
SNIP_rigl_HP_S   = [57.89, 57.44, 56.46, 55.57]


# Tape on HP-S
dense_Tape_HP_S = [64.75]
LTH_Tape_HP_S = []
GMP_Tape_HP_S = []
OMP_after_Tape_HP_S   = [65.18, 64.66, 64.48, 64.11, 64.49, 61.27, 60.06, 58.73]
OMP_before_Tape_HP_S   = [43.77, 62.12, 43.78, 61.11, 60.08, 59.57, 43.78, 43.78]
OMP_before_rigl_Tape_HP_S   = []
RP_after_Tape_HP_S   = [60.90, 60.16, 29.21, 58.17, 57.45, 56.65, 55.77, 55.22, 54.65]
RP_before_rigl_Tape_HP_S = []
SNIP_Tape_HP_S   = []
SNIP_rigl_Tape_HP_S   = []

# ESM-1b on Atlas
dense_ESM1b_atlas = [50.31]
LTH_ESM1b_atlas=                [50.91, 44.33, 40.31, 38.53, 40.35, 41.24, 40.62, ]
GMP_ESM1b_atlas =               [49.40, 49.02, 41.64, 41.32, 37.67, 39.68, 40.13, 38.04, 39.6]
OMP_after_ESM1b_atlas   =       [50.91, 45.38, 40.99, 40.29, 40.56, 40.56, 40.59, 37.54, 40.02]
OMP_before_ESM1b_atlas   =      [50.42, 43.96, 40.74, 40.48, 38.40, 40.56, 39.40, 37.41, 40.26]
OMP_before_rigl_ESM1b_atlas   = [46.51, 41.87, 40.61, 40.48, 41.00, 41.61, 41.09, 40.69, 39.75]
RP_after_ESM1b_atlas   =        [43.72, 41.50, 40.40, 40.65, 40.56, 40.41, 40.88, 39.55, 39.57]
RP_before_rigl_ESM1b_atlas =    [41.91, 41.37, 39.98, 40.04, 40.56, 41.63, 41.39, 41.15, 39.60]
SNIP_ESM1b_atlas   =            [45.31, 44.90, 44.08, 43.16, 42.55, 41.25, 41.38, 41.02, 40.41]
SNIP_rigl_ESM1b_atlas   =       [47.84, 46.57, 45.15, 42.65, 41.61, 40.90, 40.37, 40.33, 40.69]



roberta_large = fig.add_subplot(2,3,4)
roberta_large.plot(range(len(x_axis)), dense_ESM1b_HP_S*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(SNIP_HP_S)), SNIP_HP_S,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(LTH_HP_S)), LTH_HP_S,   '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(SNIP_rigl_HP_S)), SNIP_rigl_HP_S,   '--o',   label='SNIP+RIGL (Before)',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(range(len(OMP_after_HP_S)), OMP_after_HP_S,   '-o',   label='OMP (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(RP_after_HP_S)), RP_after_HP_S,   '-o',   label='Random (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(RP_before_rigl_HP_S)), RP_before_rigl_HP_S,   '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(range(len(GMP_HP_S)), GMP_HP_S,   '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(range(len(OMP_before_HP_S)), OMP_before_HP_S,   '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(OMP_before_rigl_HP_S)), OMP_before_rigl_HP_S,   '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('ESM-1b on HP-S',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.axes.get_yaxis().set_visible(True)
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



roberta_large = fig.add_subplot(2,3,5)
roberta_large.plot(range(len(x_axis)), dense_Tape_HP_S*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(range(len(RP_after_HP_S)), np.array(acc_imp_wino[1:]),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(OMP_after_Tape_HP_S)), np.array(OMP_after_Tape_HP_S),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(range(len(OMP_after_Tape_HP_S)), np.array(acc_random_after_wino),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(range(len(OMP_after_Tape_HP_S)), np.array(acc_gmp_wino),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )


# roberta_large.plot(range(len(OMP_after_Tape_HP_S)), np.array(acc_snip_wino),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(range(len(OMP_after_Tape_HP_S)), np.array(acc_snip_rigl_wino),  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
# roberta_large.plot(range(len(RP_after_Tape_HP_S)), np.array(RP_after_Tape_HP_S),  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(RP_after_Tape_HP_S)), np.array(RP_after_Tape_HP_S),  '--o', color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(range(len(OMP_before_Tape_HP_S)), np.array(OMP_before_Tape_HP_S),  '-o',  color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(range(len(OMP_after_Tape_HP_S)), np.array(acc_omp_rigl_wino),  '--o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize )



roberta_large.set_title(' Tape on HP-S+',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
roberta_large.axes.get_yaxis().set_visible(True)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



roberta_large = fig.add_subplot(2,3,6)

roberta_large.plot(range(len(x_axis)), dense_ESM1b_atlas*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(LTH_ESM1b_atlas)), LTH_ESM1b_atlas,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(OMP_after_ESM1b_atlas)), OMP_after_ESM1b_atlas,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(OMP_before_ESM1b_atlas)), OMP_before_ESM1b_atlas,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(GMP_ESM1b_atlas)), GMP_ESM1b_atlas,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )


roberta_large.plot(range(len(SNIP_ESM1b_atlas)), SNIP_ESM1b_atlas,  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(SNIP_rigl_ESM1b_atlas)), SNIP_rigl_ESM1b_atlas,  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(range(len(RP_after_ESM1b_atlas)), RP_after_ESM1b_atlas,  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(RP_before_rigl_ESM1b_atlas)), RP_before_rigl_ESM1b_atlas, '--o', color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(range(len(OMP_before_ESM1b_atlas)), OMP_before_ESM1b_atlas,  '-o', color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(range(len(OMP_before_rigl_ESM1b_atlas)), OMP_before_rigl_ESM1b_atlas,  '--o', color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('ESM-1b on Atlas',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )

# vgg_all.set_yticks([16,10,0,-10,-20,-30,-40])
# vgg_all.set_yticklabels(['',10,0,-10,-20,-30,-40],fontsize=fontsize )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.axes.get_yaxis().set_visible(True)
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=5, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.06, bottom=0.27, right=0.95, top=0.94, wspace=0.3, hspace=0.35)
# roberta_large.set_title('Roberta large on CommonsenseQA',fontsize=fontsize)
# roberta_large.set_xticks(range(10))
# roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
#
# roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
plt.savefig('Protein_sequence.pdf')
plt.show()