# This generates mock redshifts and corresponding magnitudes based on the model
# get output by calling get_data() which gives np array with a[0] as z and a[1] as mags

import numpy as np
from scipy.stats import skewnorm


def get_data(model, cosmoparams):
    n = 100

    # generate SN and apply model
    z = np.random.random_sample([n]) * 0.9 + 0.1
    dist_mod = np.array([model(i, cosmoparams) for i in z])
    # adding noise
    dist_mod += noise_skew(n)

    data = np.array([z, dist_mod])

    sigma_z = np.ones(n) * 0.001
    sigma_dist_mod = np.random.normal(0.1, 0.02, n)
    sigma_data = np.array([sigma_z, sigma_dist_mod])

    return data, sigma_data


def get_data_malmquist(model, cosmoparams):
    n = 100

    # generate SN and apply model
    z = np.random.random_sample([n]) * 0.9 + 0.1
    dist_mod = np.array([model(i, cosmoparams) for i in z])
    # adding noise
    dist_mod += noise_normal(n)

    data = np.array([z, dist_mod])

    sigma_z = np.ones(n) * 0.001
    sigma_dist_mod = np.random.normal(0.1, 0.02, n)
    sigma_data = np.array([sigma_z, sigma_dist_mod])

    condition = cut_malmquist(data, model, cosmoparams)
    cut_data = data[:, condition]
    cut_sigma_data = sigma_data[:, condition]

    return data, sigma_data, cut_data, cut_sigma_data
# Types of uncertainty on the magntiudes


def noise_normal(n):
    centre = 0.0
    sigma = 0.1  # mag
    noise = np.random.normal(centre, sigma, n)
    return noise


def noise_skew(n):
    noise = skewnorm.rvs(5, size=n)
    noise = noise - np.mean(noise)
    noise = noise * 0.1
    return noise


def distance_from_dist_mod(dist_mod):
    luminosity_distance = 10 * 10**(dist_mod/5)
    return luminosity_distance


def app_mag_from_dist_mod(dist_mod):
    M_B = -19
    app_mag = dist_mod - M_B
    return app_mag


def cut_malmquist(data, model, params):
    delta_mu = data[1] - np.array([model(i, params) for i in data[0]])
    condition = (delta_mu > 0.05) & (data[0] > 0.5)
    return ~condition
