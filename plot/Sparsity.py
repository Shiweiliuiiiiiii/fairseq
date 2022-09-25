import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 10), dpi=150, facecolor='w', edgecolor='k')
fontsize = 14
Titlesize = 18
markersize = 7
linewidth = 2.2


CSQA = torch.load('/Users/liushiwei/Projects/fairseq/results/reasoning/sparsity/CSQA_sparsity.pt')

print(CSQA)



# acc_test_omp_random_after_RACE = []
# acc_test_omp_random_after_high_RACE = []
# with open('/Users/liushiwei/Projects/fairseq/results/reasoning/race/cobert_race_omp_random_after_test.out') as file:
#     for line in file:
#         if 'test | valid on' in line:
#             acc_test_omp_random_after_RACE.append(float(line.split()[19]))
#         if ' | test1 | valid on' in line:
#             acc_test_omp_random_after_high_RACE.append(float(line.split()[19]))



# roberta_large = fig.add_subplot(1,3,1)
# roberta_large.plot(x_axis, dense_csqa*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
#
#
# roberta_large.plot(x_axis, robert_snip_csqa,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_lth_csqa,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
#
# roberta_large.plot(x_axis, robert_snip_rigl_csqa,  '--o',   label='SNIP+RIGL (Before)',color='#228B22',linewidth=linewidth, markersize=markersize )
# roberta_large.plot(x_axis, robert_gm_after_csqa,  '-o',   label='One-Shot LRR (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
#
#
# roberta_large.plot(x_axis, robert_random_before_csqa,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_random_after_csqa,  '-o',   label='Random (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
#
# roberta_large.plot(x_axis, robert_random_rigl_csqa,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, robert_gmp_csqa,  '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
#
# roberta_large.plot(x_axis, robert_gm_before_csqa,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_rigl_csqa,  '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )
#
#
# roberta_large.set_title('Roberta on CommonsenseQA',fontsize=Titlesize)
# roberta_large.axes.get_xaxis().set_visible(True)
# roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
#
# # print((dense_csqa[0] - robert_lth_csqa[3])/  dense_csqa)[]
# roberta_large.xaxis.set_ticks(x_axis)
# roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
#
# roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# # xposition = [3,7]
# # for xc in xposition:
# #     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
# roberta_large.grid(True, linestyle='-', linewidth=0.5, )
#
# roberta_large.spines['right'].set_visible(False)
# roberta_large.spines['top'].set_visible(False)
#
#
#
#
# roberta_large = fig.add_subplot(1,3,2)
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
#
# roberta_large = fig.add_subplot(2,2,3)
#
# roberta_large.plot(x_axis, [acc_test_imp_RACE[0]]*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_imp_RACE[1:],  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_omp_after_RACE,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_random_after_RACE,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_gmp_RACE,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
#
#
# roberta_large.plot(x_axis, acc_test_snip_RACE,  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_snip_rigl_RACE,  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
# roberta_large.plot(x_axis, acc_test_random_RACE,  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_random_rigl_RACE, '--o', color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, acc_test_omp_RACE,  '-o', color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_omp_rigl_RACE,  '--o', color='#bcbd22',linewidth=linewidth, markersize=markersize )
#
# print((acc_test_imp_RACE[0] - acc_test_imp_RACE[4])/  acc_test_imp_RACE[0])
#
# roberta_large.set_title('Roberta on RACE (Middle)',fontsize=Titlesize)
# roberta_large.set_xticks(range(10))
# roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
# roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
# roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
# roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
#
# # vgg_all.set_yticks([16,10,0,-10,-20,-30,-40])
# # vgg_all.set_yticklabels(['',10,0,-10,-20,-30,-40],fontsize=fontsize )
# roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# # xposition = [3,7]
# # for xc in xposition:
# #     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
# roberta_large.grid(True, linestyle='-', linewidth=0.5, )
#
# roberta_large.spines['right'].set_visible(False)
# roberta_large.spines['top'].set_visible(False)
#
#
# roberta_large = fig.add_subplot(1,3,3)
#
# roberta_large.plot(x_axis, [acc_test_imp_high_RACE[0]]*10,  '-o', color='black',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_imp_high_RACE[1:],  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_random_after_high_RACE,  '-o',   color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_omp_after_high_RACE,  '-o',   color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_gmp_high_RACE,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
#
#
# roberta_large.plot(x_axis, acc_test_snip_high_RACE,  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_snip_rigl_high_RACE,  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
# roberta_large.plot(x_axis, acc_test_random_high_RACE,  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_random_rigl_high_RACE,  '--o', color='brown',linewidth=linewidth, markersize=markersize)
# roberta_large.plot(x_axis, acc_test_omp_high_RACE,  '-o', color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, acc_test_omp_rigl_high_RACE,  '--o', color='#bcbd22',linewidth=linewidth, markersize=markersize )
#
#
# print((acc_test_imp_high_RACE[0] - acc_test_imp_high_RACE[4])/  acc_test_imp_high_RACE[0])
#
# roberta_large.set_title('Roberta on RACE (High)',fontsize=Titlesize)
# roberta_large.set_xticks(range(10))
# # roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
# roberta_large.set_xlabel('Sparsity',fontsize=Titlesize)
# roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
# roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
#
# # vgg_all.set_yticks([16,10,0,-10,-20,-30,-40])
# # vgg_all.set_yticklabels(['',10,0,-10,-20,-30,-40],fontsize=fontsize )
# roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# # xposition = [3,7]
# # for xc in xposition:
# #     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
# roberta_large.grid(True, linestyle='-', linewidth=0.5, )
#
# roberta_large.spines['right'].set_visible(False)
# roberta_large.spines['top'].set_visible(False)



# plt.tight_layout()
# fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=6, fontsize=fontsize, frameon=False)
# fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.95, wspace=0.2, hspace=0.35)
# # roberta_large.set_title('Roberta large on CommonsenseQA',fontsize=fontsize)
# # roberta_large.set_xticks(range(10))
# # roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
# #
# # roberta_large.set_ylabel('Accuracy', fontsize=fontsize)
# # roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
# plt.savefig('Roberta_commonsense_reasoning.pdf')
# plt.show()