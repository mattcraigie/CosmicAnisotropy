# This generates mock redshifts and corresponding magnitudes based on the model
# get output by calling get_data() which gives np array with a[0] as z and a[1] as mags

import numpy as np


def get_data(model, cosmoparams):
    n = 100

    # generate SN and apply model
    z = np.random.random_sample([n]) * 0.9 + 0.1
    dist_mod = np.array([model(i, cosmoparams) for i in z])
    # adding noise
    dist_mod += noise_normal(n)

    data = np.array([z, dist_mod])

    sigma_z = np.ones(n) * 0.001
    sigma_dist_mod = np.random.normal(0.1, 0.05, n)
    sigma_data = np.array([sigma_z, sigma_dist_mod])

    return data, sigma_data

# Types of uncertainty on the magntiudes


def noise_normal(n):
    centre = 0.0
    sigma = 0.1  # mag
    noise = np.random.normal(centre, sigma, n)
    return noise


def noise_skew(n):
    return None


def distance_from_dist_mod(dist_mod):
    luminosity_distance = 10 * 10**(dist_mod/5)
    return luminosity_distance

def app_mag_from_dist_mod:
    M_B = -19
    app_mag = mu - M_B
    return app_mag
