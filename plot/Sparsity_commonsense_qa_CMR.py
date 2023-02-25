import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 5), dpi=150, facecolor='w', edgecolor='k')
fontsize = 12
Titlesize = 12
markersize = 1
linewidth = 1

removed_layer = ['model.encoder.sentence_encoder.layers.0.in_proj_weight', 'model.encoder.sentence_encoder.layers.0.out_proj_weight', \
                 'model.encoder.sentence_encoder.layers.0.fc1_weight', 'model.encoder.sentence_encoder.layers.0.fc2_weight', 'model.encoder.lm_head.dense.weight']

csqa_sparsity = []
full_name = []
with open('/Users/shiwei/Projects/fairseq/results/reasoning/sparsity/cobert_race_sparsity.out') as file:
    for line in file:
        if 'sparsity of' in line:
            single_name = line.split()[-3]

            if single_name not in removed_layer:
                csqa_sparsity.append(float(line.split()[-1]))
                if 'encoder.' in single_name:
                    full_name.append(single_name.split('encoder.')[-1])
                else:
                    full_name.append(single_name)

csqa_sparsity = np.array(csqa_sparsity).reshape(7, -1)
csqa_sparsity_gm = csqa_sparsity[0].reshape(10,-1)
csqa_sparsity_gm_after = csqa_sparsity[1].reshape(10,-1)
csqa_sparsity_gmp = csqa_sparsity[2].reshape(10,-1)
csqa_sparsity_IMP = csqa_sparsity[3].reshape(10,-1)
csqa_sparsity_random = csqa_sparsity[4].reshape(10,-1)
csqa_sparsity_random_after = csqa_sparsity[5].reshape(10,-1)
csqa_sparsity_snip= csqa_sparsity[6].reshape(10,-1)

#0.2 0.36 0.488 0.590 0.672 0.738 0.791 0.8325 0.866 0.893
csqa_sparsity_gm_02 = csqa_sparsity_gm[0]
csqa_sparsity_gm_036 = csqa_sparsity_gm[1]


csqa_sparsity_gm_0488 = csqa_sparsity_gm[2]
csqa_sparsity_gm_059 = csqa_sparsity_gm[3]
csqa_sparsity_gm_0672 = csqa_sparsity_gm[4]
csqa_sparsity_gm_0738 = csqa_sparsity_gm[5]
csqa_sparsity_gm_0791 = csqa_sparsity_gm[6]
csqa_sparsity_gm_08325 = csqa_sparsity_gm[7]
csqa_sparsity_gm_0866 = csqa_sparsity_gm[8]
csqa_sparsity_gm_0893 = csqa_sparsity_gm[9]


length = len(csqa_sparsity_gm_02)

full_name = full_name[:length]

picked_names = ['embed_tokens.weight', 'embed_positions.weight', 'layers.1.in_proj_weight', \
                'layers.2.in_proj_weight',  'layers.3.in_proj_weight', 'layers.4.in_proj_weight', \
                'layers.5.in_proj_weight',  'layers.6.in_proj_weight', 'layers.7.in_proj_weight', \
'layers.8.in_proj_weight',  'layers.9.in_proj_weight', 'layers.10.in_proj_weight', \
'layers.11.in_proj_weight',  'layers.12.in_proj_weight', 'layers.13.in_proj_weight', \
'layers.14.in_proj_weight',  'layers.15.in_proj_weight', 'layers.16.in_proj_weight', \
'layers.17.in_proj_weight',  'layers.18.in_proj_weight', 'layers.19.in_proj_weight', \
'layers.20.in_proj_weight',  'layers.21.in_proj_weight', 'layers.22.in_proj_weight', 'layers.22.in_proj_weight', \
                'model.classification_heads.sentence_classification_head.dense.weight',]



print(full_name)
csqa_sparsity_gm_after_02 = csqa_sparsity_gm_after[0]
csqa_sparsity_gm_after_036 = csqa_sparsity_gm_after[1]
csqa_sparsity_gm_after_0488 = csqa_sparsity_gm_after[2]
csqa_sparsity_gm_after_059 = csqa_sparsity_gm_after[3]
csqa_sparsity_gm_after_0672 = csqa_sparsity_gm_after[4]
csqa_sparsity_gm_after_0738 = csqa_sparsity_gm_after[5]
csqa_sparsity_gm_after_0791 = csqa_sparsity_gm_after[6]
csqa_sparsity_gm_after_08325 = csqa_sparsity_gm_after[7]
csqa_sparsity_gm_after_0866 = csqa_sparsity_gm_after[8]
csqa_sparsity_gm_after_0893 = csqa_sparsity_gm_after[9]


csqa_sparsity_gmp_02 = csqa_sparsity_gmp[0]
csqa_sparsity_gmp_036 = csqa_sparsity_gmp[1]
csqa_sparsity_gmp_0488 = csqa_sparsity_gmp[2]
csqa_sparsity_gmp_059 = csqa_sparsity_gmp[3]
csqa_sparsity_gmp_0672 = csqa_sparsity_gmp[4]
csqa_sparsity_gmp_0738 = csqa_sparsity_gmp[5]
csqa_sparsity_gmp_0791 = csqa_sparsity_gmp[6]
csqa_sparsity_gmp_08325 = csqa_sparsity_gmp[7]
csqa_sparsity_gmp_0866 = csqa_sparsity_gmp[8]
csqa_sparsity_gmp_0893 = csqa_sparsity_gmp[9]


csqa_sparsity_IMP_02 = csqa_sparsity_IMP[0]
csqa_sparsity_IMP_036 = csqa_sparsity_IMP[1]
csqa_sparsity_IMP_0488 = csqa_sparsity_IMP[2]
csqa_sparsity_IMP_059 = csqa_sparsity_IMP[3]
csqa_sparsity_IMP_0672 = csqa_sparsity_IMP[4]
csqa_sparsity_IMP_0738 = csqa_sparsity_IMP[5]
csqa_sparsity_IMP_0791 = csqa_sparsity_IMP[6]
csqa_sparsity_IMP_08325 = csqa_sparsity_IMP[7]
csqa_sparsity_IMP_0866 = csqa_sparsity_IMP[8]
csqa_sparsity_IMP_0893 = csqa_sparsity_IMP[9]


csqa_sparsity_random_02 = csqa_sparsity_random[0]
csqa_sparsity_random_036 = csqa_sparsity_random[1]
csqa_sparsity_random_0488 = csqa_sparsity_random[2]
csqa_sparsity_random_059 = csqa_sparsity_random[3]
csqa_sparsity_random_0672 = csqa_sparsity_random[4]
csqa_sparsity_random_0738 = csqa_sparsity_random[5]
csqa_sparsity_random_0791 = csqa_sparsity_random[6]
csqa_sparsity_random_08325 = csqa_sparsity_random[7]
csqa_sparsity_random_0866 = csqa_sparsity_random[8]
csqa_sparsity_random_0893 = csqa_sparsity_random[9]


csqa_sparsity_random_after_02 = csqa_sparsity_random_after[0]
csqa_sparsity_random_after_036 = csqa_sparsity_random_after[1]
csqa_sparsity_random_after_0488 = csqa_sparsity_random_after[2]
csqa_sparsity_random_after_059 = csqa_sparsity_random_after[3]
csqa_sparsity_random_after_0672 = csqa_sparsity_random_after[4]
csqa_sparsity_random_after_0738 = csqa_sparsity_random_after[5]
csqa_sparsity_random_after_0791 = csqa_sparsity_random_after[6]
csqa_sparsity_random_after_08325 = csqa_sparsity_random_after[7]
csqa_sparsity_random_after_0866 = csqa_sparsity_random_after[8]
csqa_sparsity_random_after_0893 = csqa_sparsity_random_after[9]

csqa_sparsity_snip_02 = csqa_sparsity_snip[0]
csqa_sparsity_snip_036 = csqa_sparsity_snip[1]
csqa_sparsity_snip_0488 = csqa_sparsity_snip[2]
csqa_sparsity_snip_059 = csqa_sparsity_snip[3]
csqa_sparsity_snip_0672 = csqa_sparsity_snip[4]
csqa_sparsity_snip_0738 = csqa_sparsity_snip[5]
csqa_sparsity_snip_0791 = csqa_sparsity_snip[6]
csqa_sparsity_snip_08325 = csqa_sparsity_snip[7]
csqa_sparsity_snip_0866 = csqa_sparsity_snip[8]
csqa_sparsity_snip_0893 = csqa_sparsity_snip[9]


# # snns = ['gm', 'gm_after',  'gmp', 'IMP', 'random', 'random_after',  'snip']
# winogrande_sparsity = []
# with open('/Users/liushiwei/Projects/fairseq/results/reasoning/sparsity/cobert_winogrande_sparsity.out') as file:
#     for line in file:
#         if 'sparsity of' in line:
#             winogrande_sparsity.append(float(line.split()[-1]))
#
# winogrande_sparsity_gmp_imp_gmp = []
# with open('/Users/liushiwei/Projects/fairseq/results/reasoning/sparsity/cobert_winogrande_sparsity_imp_gmp.out') as file:
#     for line in file:
#         if 'sparsity of' in line:
#             winogrande_sparsity_gmp_imp_gmp.append(float(line.split()[-1]))
#
#
# winogrande_sparsity_gmp_imp = np.array(winogrande_sparsity_gmp_imp_gmp).reshape(2, -1)
# winogrande_sparsity_gmp = winogrande_sparsity_gmp_imp[0].reshape(3,-1)
# winogrande_sparsity_IMP = winogrande_sparsity_gmp_imp[1].reshape(3,-1)
#
#
#
# winogrande_sparsity = np.array(winogrande_sparsity).reshape(7, -1)
#
# winogrande_sparsity_gm = winogrande_sparsity[0].reshape(3,-1)
# winogrande_sparsity_gm_after = winogrande_sparsity[1].reshape(3,-1)
# # winogrande_sparsity_gmp = winogrande_sparsity[2].reshape(3,-1)
# # winogrande_sparsity_IMP = winogrande_sparsity[3].reshape(3,-1)
# winogrande_sparsity_random = winogrande_sparsity[4].reshape(3,-1)
# winogrande_sparsity_random_after = winogrande_sparsity[5].reshape(3,-1)
# winogrande_sparsity_snip= winogrande_sparsity[6].reshape(3,-1)
#
# winogrande_sparsity_gm_36 = winogrande_sparsity_gm[0]
# winogrande_sparsity_gm_672 = winogrande_sparsity_gm[1]
# winogrande_sparsity_gm_832 = winogrande_sparsity_gm[2]
#
# winogrande_sparsity_gm_after_36 = winogrande_sparsity_gm_after[0]
# winogrande_sparsity_gm_after_672 = winogrande_sparsity_gm_after[1]
# winogrande_sparsity_gm_after_832 = winogrande_sparsity_gm_after[2]
#
# winogrande_sparsity_gmp_36 = winogrande_sparsity_gmp[0]
# winogrande_sparsity_gmp_672 = winogrande_sparsity_gmp[1]
# winogrande_sparsity_gmp_832 = winogrande_sparsity_gmp[2]
#
# winogrande_sparsity_IMP_36 = winogrande_sparsity_IMP[0]
# winogrande_sparsity_IMP_672 = winogrande_sparsity_IMP[1]
# winogrande_sparsity_IMP_832 = winogrande_sparsity_IMP[2]
# print(winogrande_sparsity_IMP_36)
# winogrande_sparsity_random_36 = winogrande_sparsity_random[0]
# winogrande_sparsity_random_672 = winogrande_sparsity_random[1]
# winogrande_sparsity_random_832 = winogrande_sparsity_random[2]
#
# winogrande_sparsity_random_after_36 = winogrande_sparsity_random_after[0]
# winogrande_sparsity_random_after_672 = winogrande_sparsity_random_after[1]
# winogrande_sparsity_random_after_832 = winogrande_sparsity_random_after[2]
#
# winogrande_sparsity_snip_36 = winogrande_sparsity_snip[0]
# winogrande_sparsity_snip_672 = winogrande_sparsity_snip[1]
# winogrande_sparsity_snip_832 = winogrande_sparsity_snip[2]
#
# x_axis = range(len(winogrande_sparsity_gm_36))




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

csqa_sparsity_LTH_noemb_nocla = []
full_name_LTH_noemb_nocla = []
with open('/Users/shiwei/Projects/fairseq/roberta_LTH_noembed_noclassifier.out') as file:
    for line in file:
        if 'sparsity of' in line and 'torch.Size' in line:
            single_name = line.split()[-3]

            if single_name not in removed_layer:
                csqa_sparsity_LTH_noemb_nocla.append(float(line.split()[-1]))
                if 'encoder.' in single_name:
                    full_name_LTH_noemb_nocla.append(single_name.split('encoder.')[-1])

                else:
                    full_name_LTH_noemb_nocla.append(single_name)

# print(len(np.array(csqa_sparsity_LTH_LRR_noemb_nocla).reshape(10, -1)[1]))


csqa_sparsity_LTH_LRR_noemb_nocla = []
full_name_LTH_LRR_noemb_nocla = []
with open('/Users/shiwei/Projects/fairseq/roberta_LTH_LLR_noembed_noclassifier.out') as file:
    for line in file:
        if 'sparsity of' in line and 'torch.Size' in line:
            single_name = line.split()[3]

            if single_name not in removed_layer:
                csqa_sparsity_LTH_LRR_noemb_nocla.append(float(line.split()[-1]))
                if 'encoder.' in single_name:
                    full_name_LTH_LRR_noemb_nocla.append(single_name.split('encoder.')[-1])
                else:
                    full_name_LTH_LRR_noemb_nocla.append(single_name)

print(len(np.array(csqa_sparsity_LTH_LRR_noemb_nocla).reshape(10, -1)[1]))


csqa_sparsity_oBERT_KD_LRR_noemb_nocla = []
full_name_oBERT_LRR_noemb_nocla = []
with open('/Users/shiwei/Projects/fairseq/roberta_oBERT_LLR_KD_noembed_noclassifier.txt') as file:
    for line in file:
        if 'sparsity of' in line and 'torch.Size' in line:
            single_name = line.split()[-3]

            if single_name not in removed_layer:
                csqa_sparsity_oBERT_KD_LRR_noemb_nocla.append(float(line.split()[-1]))
                if 'encoder.' in single_name:
                    full_name_oBERT_LRR_noemb_nocla.append(single_name.split('encoder.')[-1])
                else:
                    full_name_oBERT_LRR_noemb_nocla.append(single_name)

# csqa_sparsity_oBERT_KD_LRR_noemb_nocla = len(np.array(csqa_sparsity_oBERT_KD_LRR_noemb_nocla).reshape(10, -1)[1])

# full_name_LTH_noemb_nocla = full_name_LTH_noemb_nocla[:144]
# print(np.array(full_name_LTH_LRR_noemb_nocla).reshape(10, -1)[1])


csqa_sparsity_oBERT_LLR_noemb_nocla = []
full_name_oBERT_LLR_noemb_nocla = []
with open('/Users/shiwei/Projects/fairseq/roberta_oBERT_LLR_noembed_noclassifier.out') as file:
    for line in file:
        if 'sparsity of' in line and 'torch.Size' in line:
            single_name = line.split()[-3]
            if single_name not in removed_layer:
                csqa_sparsity_oBERT_LLR_noemb_nocla.append(float(line.split()[-1]))
                if 'encoder.' in single_name:
                    full_name_oBERT_LLR_noemb_nocla.append(single_name.split('encoder.')[-1])
                else:
                    full_name_oBERT_LLR_noemb_nocla.append(single_name)

# csqa_sparsity_oBERT_LLR_noemb_nocla = len(np.array(csqa_ sparsity_oBERT_LLR_noemb_nocla).reshape(10, -1)[1])

# full_name_LTH_noemb_nocla = full_name_LTH_noemb_nocla[:144]
# print(np.array(full_name_LTH_LRR_noemb_nocla).reshape(10, -1)[1])


full_name_LTH_LRR_noemb_nocla = full_name_LTH_LRR_noemb_nocla[:144]

# print(full_name_LTH_LRR_noemb_nocla)
for i in range(len(full_name_LTH_LRR_noemb_nocla)):
    # if 'self_attn' in full_name_LTH_LRR_noemb_nocla[i]:
    #     full_name_LTH_LRR_noemb_nocla[i] = ''
    if 'layers' in full_name_LTH_LRR_noemb_nocla[i]:
        full_name_LTH_LRR_noemb_nocla[i] = full_name_LTH_LRR_noemb_nocla[i].replace('layers', 'L')
    elif 'sentence_classification_head' in full_name_LTH_LRR_noemb_nocla[i]:
        full_name_LTH_LRR_noemb_nocla[i] = 'classification_heads'

# full_name_LTH_LRR_noemb_nocla[1] = ''
full_name_LTH_LRR_noemb_nocla[2] = 'embed_positions.weight'
print(full_name_LTH_LRR_noemb_nocla)


x_axis = range(144)
roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, np.array(csqa_sparsity_LTH_noemb_nocla).reshape(10, -1)[1],  '-o',  label='LTH (not pruning emb and classifier)', color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_LTH_LRR_noemb_nocla).reshape(10, -1)[1],  '-o',  label='LTH + LRR (not pruning emb and classifier)', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_oBERT_KD_LRR_noemb_nocla).reshape(10, -1)[1],  '-o',  label='oBERT + LRR + KD (not pruning emb and classifier)', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, np.array(csqa_sparsity_oBERT_LLR_noemb_nocla).reshape(10, -1)[1],  '-o',  label='oBERT + LRR (not pruning emb and classifier)', color='pink',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_036),  '--o',  label='OMP (Before)', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, np.array(csqa_sparsity_random_after_36),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gmp_036   ),  '-o', label='GMP (During)', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_snip_036),  '-o',  label='SNIP (Before)', color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_random_036),  '-o', label='Random (Before)', color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_after_036),  '-o', label='OMP (After)', color='#00BFFF',linewidth=linewidth, markersize=markersize, )

# roberta_large.set_ylabel('Layerwise Sparsity', fontsize=fontsize)
# roberta_large.set_title('RoBERTa on RACE, Overall Sparity=36%',fontsize=Titlesize)
# roberta_large.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labelsize=fontsize)
# roberta_large.xaxis.set_ticks(x_axis, rotation=45)
# roberta_large.set_xticklabels(['']*len(x_axis), rotation=90, fontsize=1)
# roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# roberta_large.axes.get_xaxis().set_visible(False)
# roberta_large.grid(True, axis='y',  linestyle='-', linewidth=0.5, )
# roberta_large.spines['right'].set_visible(False)
# roberta_large.spines['top'].set_visible(False)


# x_axis = range(len(full_name))
# roberta_large = fig.add_subplot(3,1,1)
# roberta_large.plot(x_axis, np.array(csqa_sparsity_IMP_036),  '-o',  label='LTH (After)', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_036),  '--o',  label='OMP (Before)', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, np.array(csqa_sparsity_random_after_36),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gmp_036   ),  '-o', label='GMP (During)', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_snip_036),  '-o',  label='SNIP (Before)', color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_random_036),  '-o', label='Random (Before)', color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_after_036),  '-o', label='OMP (After)', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
#
#
#
# roberta_large.set_ylabel('Layerwise Sparsity', fontsize=fontsize)
# roberta_large.set_title('RoBERTa on RACE, Overall Sparity=36%',fontsize=Titlesize)
# roberta_large.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labelsize=fontsize)
# roberta_large.xaxis.set_ticks(x_axis, rotation=45)
# roberta_large.set_xticklabels(['']*len(x_axis), rotation=90, fontsize=1)
# roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# roberta_large.axes.get_xaxis().set_visible(False)
# roberta_large.grid(True, axis='y',  linestyle='-', linewidth=0.5, )
# roberta_large.spines['right'].set_visible(False)
# roberta_large.spines['top'].set_visible(False)


#
# x_axis = range(len(full_name))
# roberta_large = fig.add_subplot(3,1,2)
# roberta_large.plot(x_axis, np.array(csqa_sparsity_IMP_0672),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_0672),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, np.array(csqa_sparsity_random_after_672),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gmp_0672),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_snip_0672),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_random_0672),  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_after_0672),  '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )
#
#
#
# roberta_large.set_ylabel('Layerwise Sparsity', fontsize=fontsize)
# roberta_large.set_title('Roberta on RACE, Overall Sparity=67%',fontsize=Titlesize)
# roberta_large.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labelsize=fontsize)
# roberta_large.xaxis.set_ticks(x_axis, rotation=45)
# roberta_large.set_xticklabels(['']*len(x_axis), rotation=90, fontsize=1)
# roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# roberta_large.axes.get_xaxis().set_visible(False)
# roberta_large.grid(True, axis='y',  linestyle='-', linewidth=0.5, )
# roberta_large.spines['right'].set_visible(False)
# roberta_large.spines['top'].set_visible(False)
#
#
# x_axis = range(len(full_name))
# roberta_large = fig.add_subplot(3,1,3)
# roberta_large.plot(x_axis, np.array(csqa_sparsity_IMP_08325),  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_08325),  '-o', color='#00BFFF',linewidth=linewidth, markersize=markersize, )
# # roberta_large.plot(x_axis, np.array(csqa_sparsity_random_after_832),  '-o', color='#0072BD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gmp_08325),  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_snip_08325),  '-o', color='#228B22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_random_08325),  '-o', color='brown',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, np.array(csqa_sparsity_gm_after_08325),  '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )
#


roberta_large.set_ylabel('Layerwise Sparsity', fontsize=fontsize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.set_title('RoBERTa on CSQA, Overall Sparity=36%',fontsize=Titlesize)
roberta_large.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labelsize=fontsize)
roberta_large.xaxis.set_ticks(x_axis, rotation=90, fontsize=1)
roberta_large.set_xticklabels(full_name_LTH_LRR_noemb_nocla[:144], rotation=90, fontsize=3)
# roberta_large.tick_params(axis='both', which='major', labelsize=5)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.grid(True, axis='y', linestyle='-', linewidth=0.5, )
roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)

plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=2, fontsize=8, frameon=False)
fig.subplots_adjust(left=0.05, bottom=0.29, right=0.95, top=0.94, wspace=0.2, hspace=0.35)

plt.savefig('layerwise_sparsity_winogrande_roadmap.png')
plt.show()