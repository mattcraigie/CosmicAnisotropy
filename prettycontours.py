import numpy as np
from getdist import loadMCSamples
from getdist import plots, MCSamples
import matplotlib.pylab as plt

from astropy.cosmology import FlatwCDM

defaultsmoothing = {}
reducedsmoothing = {'mult_bias_correction_order': 0, 'smooth_scale_2D': 0.1, 'smooth_scale_1D': 0.1}

regions = ['C', 'E', 'S', 'X']
fpath = './sep_fields_plus_lowz/cosmomc_chains/{}_chains/nohubble_sn_omw_0'
region_samples = [loadMCSamples(fpath.format(l.lower()), settings=reducedsmoothing) for l in regions]
all_samples = loadMCSamples(fpath.format('all'), settings=reducedsmoothing)

tempdir = '.'

[print(a.likeStats) for a in region_samples]
exit()

g = plots.get_subplot_plotter(width_inch=10, subplot_size_ratio=1)

# settings
g.settings.legend_fontsize = 18
g.settings.alpha_filled_add = 0.4

# individual regions
# g.plot_2d(region_samples, param_pair=['omegam', 'w'], filled=True, lims=[0, 0.7, -2, -0.3],
#           colors=[('royalblue'), ('red'), ('goldenrod'), ('deeppink')], nx=2)

g.plots_2d_triplets(root_params_triplets=[(i, 'omegam', 'w') for i in region_samples], filled=True, nx=2)


# fix the axes and add an extra 'all' contour every axis
# for i in [0, 1]:
#     for j in [0, 1]:
#         ax = g.get_axes(i, j)
#         ax.set_xlim(0.0, 0.7)
#         ax.set_ylim(-0.3, -2)

for i in [0, 1]:
    for j in [0, 1]:
        g.add_2d_contours(all_samples, 'omegam', 'w', filled=False, ls='--', ax=[i, j])
        g.set_axes(lims=[0.0, 0.7, -2, -0.3], ax=[i, j])



for i in [0,1,2,3]:
    ax = g.get_axes(i)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.text(0.5, 0.92, s=regions[i], fontsize=20, horizontalalignment='center', transform=ax.transAxes)
    if i in [0, 1]:
        ax.set_xticklabels([])
    if i in [1, 3]:
        ax.set_yticklabels([])

g.finish_plot()

g.export('pretty_wcdm_contours.png')


# bestparams = samples1.getParamBestFitDict(best_sample=True)
# print(bestparams)