# This generates mock redshifts and corresponding magnitudes based on the model
# get output by calling get_data() which gives np array with a[0] as z and a[1] as mags

import numpy as np


def get_data(model, cosmoparams):
    n = 100

    # generate SN and apply model
    z = np.random.random_sample([n]) * 0.9 + 0.1
    mags = np.array([model(i, cosmoparams) for i in z])
    # adding noise
    mags += noise_normal(n)

    data = np.array([z, mags])

    sigma_z = np.ones(n) * 0.001
    sigma_mags = np.random.normal(0.1, 0.05, n)
    sigma_data = np.array([sigma_z, sigma_mags])

    return data, sigma_data

# Types of uncertainty on the magntiudes


def noise_normal(n):
    centre = 0.0
    sigma = 0.1  # mag
    noise = np.random.normal(centre, sigma, n)
    return noise

def noise_skew(n):
    return None
