# This generates mock redshifts and corresponding magnitudes based on the model
# get output by calling get_data() which gives np array with a[0] as z and a[1] as mags

import numpy as np


def get_data(model, cosmoparams):
    n = 100

    # generate SN and apply model
    z = np.random.random_sample([n])
    mags = np.array([model(i, cosmoparams) for i in z])
    print(mags)
    # adding noise
    sigma = 0.05  # mag
    noise = np.random.normal(0.01, sigma, n)
    mags += noise

    data = np.array([z, mags])
    return data
