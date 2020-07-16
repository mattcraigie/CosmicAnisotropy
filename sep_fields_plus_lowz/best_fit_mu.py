import numpy as np
from chainconsumer import ChainConsumer


c = ChainConsumer()
c.add_chain('(SN + PRIOR) ALL_XFIELD_ALL.csv.gz')
fig = c.plotter.plot_walks(truth={"$x$": -1, "$y$": 1, "$z$": -2}, convolve=100)

fig.savefig('walk.png')