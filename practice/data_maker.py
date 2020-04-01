# This generates mock redshifts and corresponding magnitudes based on the model
# get output by calling get_data() which gives np array with a[0] as z and a[1] as mags

import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

def get_data(model, cosmoparams):
    n = 100

    # generate SN and apply model
    z = np.random.random_sample([n]) * 0.9 + 0.1
    dist_mod = np.array([model(i, cosmoparams) for i in z])
    # adding noise
    dist_mod += noise_skew(n)

    data = np.array([z, dist_mod])

    sigma_z = np.ones(n) * 0.001
    sigma_dist_mod = np.random.normal(0.1, 0.05, n)
    sigma_data = np.array([sigma_z, sigma_dist_mod])

    return data, sigma_data

# Types of uncertainty on the magnitudes


def noise_normal(n):
    centre = 0.0
    sigma = 0.1  # mag
    noise = np.random.normal(centre, sigma, n)
    return noise


def noise_skew(n):
    centre = 0.0
    sigma = 0.1
    a = -1
    noise = skewnorm.rvs(a, size=n*1000, loc=0, scale=1)
    plt.hist(noise, bins=30)
    plt.show()
    return noise


def distance_from_dist_mod(dist_mod):
    luminosity_distance = 10 * 10**(dist_mod/5)
    return luminosity_distance


def app_mag_from_dist_mod(dist_mod):
    abs_mag_b = -19
    app_mag_b = dist_mod - abs_mag_b
    return app_mag_b
