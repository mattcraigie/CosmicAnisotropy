import numpy as np
from getdist import loadMCSamples
from getdist import plots, MCSamples
import matplotlib.pylab as plt

from astropy.cosmology import FlatwCDM

samples1 = loadMCSamples('./sn_prior_omw_0')
tempdir = '.'

# g = plots.getSubplotPlotter()
# g.triangle_plot([samples1], filled=True, params=['omegam', 'w', 'H0'])
#
# g = plots.get_single_plotter(width_inch=5)
# g.plot_3d(samples1, ['omegam', 'w', 'H0'])

# g.export('sn_prior_omw.png')


bestparams = samples1.getParamBestFitDict(best_sample=True)
print(bestparams)
H0, om, w = bestparams['H0'], bestparams['omegam'], bestparams['w']

zrange = np.linspace(0.01, 1)
distmod_fit = FlatwCDM(H0=H0, Om0=om, w0=w).distmod(zrange).value
distmod_real = FlatwCDM(H0=70, Om0=om, w0=w).distmod(zrange).value

plt.plot(zrange, distmod_fit, c='red', linestyle=':')
plt.plot(zrange, distmod_real, c='black')

plt.savefig('compare.png')