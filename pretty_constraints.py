import numpy as np
from getdist import loadMCSamples
from getdist import plots, MCSamples
import matplotlib.pylab as plt

from astropy.cosmology import FlatwCDM

letters = ['ALL', 'C', 'E', 'S', 'X']
fpath = './sep_fields_plus_lowz/cosmomc_chains/{}_chains/nohubble_sn_omw_0'
samples = [loadMCSamples(fpath.format(l.lower())) for l in letters]

om_means = [s.getMeans()[0] for s in samples]
om_variances = [s.getVars()[0] for s in samples]


fig, ax = plt.subplots()

y_pos = np.linspace(0.5, 0.1, 5)

for l, m, v, y in zip(letters, om_means, om_variances, y_pos):
    print(m, y, v)
    plt.scatter([m], [y], )
    ax.errorbar([m], [y], xerr=np.sqrt(v))


plt.savefig('pretty_constraints.png')