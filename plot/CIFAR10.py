import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(12, 7), dpi=90, facecolor='w', edgecolor='k')
fontsize = 15
Titlesize = 18
markersize = 7
linewidth = 2.2
x_axis = range(10)


# commonsenseQA
dense_cf10 = [92.430]
x_lth_csqa = [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]


RN20_cf10_gmp = [92.190, 92.010, 91.990, 91.440, 91.780, 91.440, 91.580, 91.350, 91.020, 91.090]
RN20_cf10_OMP_AFTER  = [92.360, 92.410, 92.290, 92.160, 91.920, 92.030, 91.430, 91.080, 90.970, 90.330]
RN20_cf10_RANDOM_AFTER   = [92.120, 90.690, 90.420, 90.170, 90.290, 89.240, 88.670, 88.420, 87.460, 86.860]
RN20_cf10_RANDOM_BEFORE  = [92.100, 91.690, 91.690, 90.950, 90.390, 89.880, 89.070, 88.690, 87.260, 86.300]
RN20_cf10_RANDOM_RIGL = [92.480, 92.100, 91.750, 91.850, 91.710, 91.070, 90.680, 89.580, 89.800, 88.530]
RN20_cf10_lth = [92.360, 92.280, 92.500, 92.210, 92.650, 92.110, 92.270, 92.110, 91.730, 91.140]
robert_snip_csqa = [92.15, 91.97, 92.20, 91.47, 91.00, 90.67, 90.56, 89.60, 88.84, 88.02]
robert_snip_rigl_csqa = [92.03, 92.10, 91.87, 91.95, 91.52, 91.24, 91.12, 90.46, 90.20, 89.16]


roberta_large = fig.add_subplot(1,1,1)
roberta_large.plot(x_axis, dense_cf10*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_csqa,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, RN20_cf10_lth,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_rigl_csqa,  '--o',   label='SNIP+RIGL (Before)',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, RN20_cf10_OMP_AFTER,  '-o',   label='OMP (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, RN20_cf10_RANDOM_BEFORE,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, RN20_cf10_RANDOM_AFTER,  '-o',   label='Random (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, RN20_cf10_RANDOM_RIGL,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, RN20_cf10_gmp,  '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )

# roberta_large.plot(x_axis, robert_gm_before_csqa,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_rigl_csqa,  '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )




roberta_large.set_title('ResNet-20 on CIFAR-10',fontsize=Titlesize)
roberta_large.axes.get_xaxis().set_visible(True)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity', fontsize=Titlesize)


roberta_large.xaxis.set_ticks(x_axis)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )

roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)
plt.ylim((80, 94))


plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=4, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.05, bottom=0.29, right=0.95, top=0.94, wspace=0.2, hspace=0.35)
plt.savefig('CIFAR-10.pdf', bbox_inches='tight')
plt.show()
