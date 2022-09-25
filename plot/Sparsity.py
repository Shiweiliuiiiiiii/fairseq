import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 10), dpi=150, facecolor='w', edgecolor='k')
fontsize = 8
Titlesize = 10
markersize = 1
linewidth = 1



csqa_sparsity = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/sparsity/cobert_qa_sparsity.out') as file:
    for line in file:
        if 'sparsity of' in line:
            csqa_sparsity.append(float(line.split()[-1]))

csqa_sparsity = np.array(csqa_sparsity).reshape(7, -1)

csqa_sparsity_gm = csqa_sparsity[0].reshape(3,-1)
csqa_sparsity_gm_after = csqa_sparsity[1].reshape(3,-1)
csqa_sparsity_gmp = csqa_sparsity[2].reshape(3,-1)
csqa_sparsity_IMP = csqa_sparsity[3].reshape(3,-1)
csqa_sparsity_random = csqa_sparsity[4].reshape(3,-1)
csqa_sparsity_random_after = csqa_sparsity[5].reshape(3,-1)
csqa_sparsity_snip= csqa_sparsity[6].reshape(3,-1)

csqa_sparsity_gm_36 = csqa_sparsity_gm[0]
csqa_sparsity_gm_672 = csqa_sparsity_gm[1]
csqa_sparsity_gm_832 = csqa_sparsity_gm[2]

csqa_sparsity_gm_after_36 = csqa_sparsity_gm_after[0]
csqa_sparsity_gm_after_672 = csqa_sparsity_gm_after[1]
csqa_sparsity_gm_after_832 = csqa_sparsity_gm_after[2]

csqa_sparsity_gmp_36 = csqa_sparsity_gmp[0]
csqa_sparsity_gmp_672 = csqa_sparsity_gmp[1]
csqa_sparsity_gmp_832 = csqa_sparsity_gmp[2]

csqa_sparsity_IMP_36 = csqa_sparsity_IMP[0]
csqa_sparsity_IMP_672 = csqa_sparsity_IMP[1]
csqa_sparsity_IMP_832 = csqa_sparsity_IMP[2]

csqa_sparsity_random_36 = csqa_sparsity_random[0]
csqa_sparsity_random_672 = csqa_sparsity_random[1]
csqa_sparsity_random_832 = csqa_sparsity_random[2]

csqa_sparsity_random_after_36 = csqa_sparsity_random_after[0]
csqa_sparsity_random_after_672 = csqa_sparsity_random_after[1]
csqa_sparsity_random_after_832 = csqa_sparsity_random_after[2]

csqa_sparsity_snip_36 = csqa_sparsity_snip[0]
csqa_sparsity_snip_672 = csqa_sparsity_snip[1]
csqa_sparsity_snip_832 = csqa_sparsity_snip[2]


# snns = ['gm', 'gm_after',  'gmp', 'IMP', 'random', 'random_after',  'snip']
winogrande_sparsity = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/sparsity/cobert_winogrande_sparsity.out') as file:
    for line in file:
        if 'sparsity of' in line:
            winogrande_sparsity.append(float(line.split()[-1]))

winogrande_sparsity_gmp_imp_gmp = []
with open('/Users/liushiwei/Projects/fairseq/results/reasoning/sparsity/cobert_winogrande_sparsity_imp_gmp.out') as file:
    for line in file:
        if 'sparsity of' in line:
            winogrande_sparsity_gmp_imp_gmp.append(float(line.split()[-1]))


winogrande_sparsity_gmp_imp = np.array(winogrande_sparsity_gmp_imp_gmp).reshape(2, -1)
winogrande_sparsity_gmp = winogrande_sparsity_gmp_imp[0].reshape(3,-1)
winogrande_sparsity_IMP = winogrande_sparsity_gmp_imp[1].reshape(3,-1)



winogrande_sparsity = np.array(winogrande_sparsity).reshape(7, -1)

winogrande_sparsity_gm = winogrande_sparsity[0].reshape(3,-1)
winogrande_sparsity_gm_after = winogrande_sparsity[1].reshape(3,-1)
# winogrande_sparsity_gmp = winogrande_sparsity[2].reshape(3,-1)
# winogrande_sparsity_IMP = winogrande_sparsity[3].reshape(3,-1)
winogrande_sparsity_random = winogrande_sparsity[4].reshape(3,-1)
winogrande_sparsity_random_after = winogrande_sparsity[5].reshape(3,-1)
winogrande_sparsity_snip= winogrande_sparsity[6].reshape(3,-1)

winogrande_sparsity_gm_36 = winogrande_sparsity_gm[0]
winogrande_sparsity_gm_672 = winogrande_sparsity_gm[1]
winogrande_sparsity_gm_832 = winogrande_sparsity_gm[2]

winogrande_sparsity_gm_after_36 = winogrande_sparsity_gm_after[0]
winogrande_sparsity_gm_after_672 = winogrande_sparsity_gm_after[1]
winogrande_sparsity_gm_after_832 = winogrande_sparsity_gm_after[2]

winogrande_sparsity_gmp_36 = winogrande_sparsity_gmp[0]
winogrande_sparsity_gmp_672 = winogrande_sparsity_gmp[1]
winogrande_sparsity_gmp_832 = winogrande_sparsity_gmp[2]

winogrande_sparsity_IMP_36 = winogrande_sparsity_IMP[0]
winogrande_sparsity_IMP_672 = winogrande_sparsity_IMP[1]
winogrande_sparsity_IMP_832 = winogrande_sparsity_IMP[2]
print(winogrande_sparsity_IMP_36)
winogrande_sparsity_random_36 = winogrande_sparsity_random[0]
winogrande_sparsity_random_672 = winogrande_sparsity_random[1]
winogrande_sparsity_random_832 = winogrande_sparsity_random[2]

winogrande_sparsity_random_after_36 = winogrande_sparsity_random_after[0]
winogrande_sparsity_random_after_672 = winogrande_sparsity_random_after[1]
winogrande_sparsity_random_after_832 = winogrande_sparsity_random_after[2]

winogrande_sparsity_snip_36 = winogrande_sparsity_snip[0]
winogrande_sparsity_snip_672 = winogrande_sparsity_snip[1]
winogrande_sparsity_snip_832 = winogrande_sparsity_snip[2]

x_axis = range(len(winogrande_sparsity_gm_36))




# # roberta_large.plot(x_axis, winogrande_sparsity_gm_36,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
#
#
# # roberta_large.plot(x_axis, winogrande_sparsity_snip_36,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, winogrande_sparsity_IMP_36,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, winogrande_sparsity_gm_after_36,  '-o',   label='One-Shot LRR (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, winogrande_sparsity_random_36,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, winogrande_sparsity_random_after_36,  '-o',   label='Random (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, winogrande_sparsity_gmp_36,  '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
#
# # roberta_large.plot(x_axis, winogrande_sparsity_gm_36,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, robert_gm_rigl_csqa,  '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )
#
#
# roberta_large.set_title('Roberta on CommonsenseQA',fontsize=Titlesize)
# roberta_large.axes.get_xaxis().set_visible(True)
# roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
#
# # print((dense_csqa[0] - robert_lth_csqa[3])/  dense_csqa)[]
# roberta_large.xaxis.set_ticks(x_axis)
# # roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
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
x_axis = range(len(csqa_sparsity_IMP_36))
roberta_large = fig.add_subplot(3,1,1)
roberta_large.plot(x_axis, np.array(csqa_sparsity_IMP_36),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_36),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_random_after_36),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_gmp_36),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_snip_36),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_random_36),  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_after_36),  '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )




roberta_large.set_title('Roberta on WinoGrande',fontsize=Titlesize)
# roberta_large.xaxis.set_ticks(x_axis, rotation=45)
# roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)



x_axis = range(len(csqa_sparsity_IMP_672))
roberta_large = fig.add_subplot(3,1,2)
roberta_large.plot(x_axis, np.array(csqa_sparsity_IMP_672),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_672),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_random_after_672),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_gmp_672),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_snip_672),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_random_672),  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_after_672),  '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )




roberta_large.set_title('Roberta on WinoGrande',fontsize=Titlesize)
# roberta_large.xaxis.set_ticks(x_axis, rotation=45)
# roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


x_axis = range(len(csqa_sparsity_IMP_832))
roberta_large = fig.add_subplot(3,1,3)
roberta_large.plot(x_axis, np.array(csqa_sparsity_IMP_832),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_832),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_random_after_832),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_gmp_832),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_snip_832),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_random_832),  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_after_832),  '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )




roberta_large.set_title('Roberta on WinoGrande',fontsize=Titlesize)
# roberta_large.xaxis.set_ticks(x_axis, rotation=45)
# roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)

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
plt.show()