from getdist import loadMCSamples
from getdist import plots, MCSamples
import matplotlib.pylab as plt
samples1 = loadMCSamples('./sn_prior_omw_0')

tempdir = '.'

g = plots.getSubplotPlotter()
g.triangle_plot([samples1], filled=True)
g.export('base_plikHM_TTTEEE_lowl_lowE_lensing.png')


print(samples1.getTable(limit=1).tableTex())
print(samples1.getTable(limit=2).tableTex())
print(samples1.getTable(limit=3).tableTex())
