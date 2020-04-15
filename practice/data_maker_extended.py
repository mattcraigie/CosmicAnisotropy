# This generates mock redshifts and corresponding magnitudes based on the model
# get output by calling get_data() which gives np array with a[0] as z and a[1] as mags

import numpy as np
from scipy.stats import skewnorm, trim_mean
import matplotlib.pyplot as plt

def get_data(model, cosmoparams):
    n = 1000

    # generate SN and apply model
    z = np.random.random_sample([n]) * 0.9 + 0.1
    mu = np.array([model(i, cosmoparams) for i in z])

    mu += noise_normal(n)  # from inherent dispersion
    mu += noise_skew(n)  # from lensing

    M0 = -19  # mean SN mag
    M_unadjusted = M0 + np.random.normal(0, 1, n)  # the adjustment factors
    m_unadjusted = mu + M_unadjusted

    # plt.hist(mu, bins=40, alpha=0.2)
    # plt.show()

    # find below qth percentile
    percentile_cut = 90
    perc = np.percentile(m_unadjusted, percentile_cut)
    cut = m_unadjusted < perc

    data = np.array([z, mu])

    sigma_z = np.ones(n) * 0.001
    sigma_dist_mod = np.random.normal(0.1, 0.02, n)
    sigma_data = np.array([sigma_z, sigma_dist_mod])

    return data, sigma_data, data[:, cut], sigma_data[:, cut]



def noise_normal(n):
    centre = 0.0
    sigma = 0.1  # mag
    noise = np.random.normal(centre, sigma, n)
    return noise


def noise_skew(n):
    noise = skewnorm.rvs(20, size=n)
    noise = noise - np.mean(noise)
    noise = noise * 0.016 * 100
    print(np.std(noise))
    return noise


def cut_malmquist(data, model, params):
    delta_mu = data[1] - np.array([model(i, params) for i in data[0]])
    condition = (delta_mu > 0.05) & (data[0] > 0.5)
    return ~condition
