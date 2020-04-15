import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm, trim_mean

import model_maker
import data_maker_extended
import hubble_diagram

np.random.seed(1234)


def chi2_fit_all(combinations, data, sigma_data, model):

    sn_z = data[0]
    sigma_mags = sigma_data[1]
    true_mags = np.array([model(i, flat_lcdm_params) for i in sn_z])

    all_chi2 = np.empty(np.shape(combinations)[0])
    print("{} calcs to do".format(np.shape(combinations)[0]))
    for i in range(np.shape(combinations)[0]):
        if i % 100 == 0:
            print(i)
        model_params = tuple(combinations[i])
        model_mags = np.array([model(i, model_params) for i in sn_z])
        chi2 = np.sum(((true_mags - model_mags) / sigma_mags) ** 2)
        all_chi2[i] = chi2

    best_fit_idx = np.argmin(all_chi2)
    best_fit_params = tuple(combinations[best_fit_idx])
    return best_fit_params


def get_data(model, cosmoparams, description):
    n = 2000

    # generate SN and apply model
    z = np.random.random_sample([n]) * 0.9 + 0.1
    mu = np.array([model(i, cosmoparams) for i in z])

    if 'gaussian' in description:
        mu += noise_normal(n)
    if 'skew' in description:
        mu += noise_skew(n)

    data = np.array([z, mu])

    sigma_z = np.ones(n) * 0.001
    sigma_dist_mod = np.random.normal(0.1, 0.02, n)
    sigma_data = np.array([sigma_z, sigma_dist_mod])


    if 'malmquist' in description:
        M0 = -19  # mean SN mag
        M_unadjusted = M0 + np.random.normal(0, 1, n)
        m_unadjusted = mu + M_unadjusted

        # find below qth percentile
        percentile_cut = 90
        perc = np.percentile(m_unadjusted, percentile_cut)
        cut = m_unadjusted < perc
        return data[:, cut], sigma_data[:, cut]
    else:
        return data, sigma_data


def noise_normal(n):
    centre = 0.0
    sigma = 0.1  # mag
    noise = np.random.normal(centre, sigma, n)
    return noise


def noise_skew(n):
    noise = skewnorm.rvs(5, size=n)
    noise = noise - np.mean(noise)
    noise = noise * 0.016
    return noise


def cut_malmquist(data, model, params):
    delta_mu = data[1] - np.array([model(i, params) for i in data[0]])
    condition = (delta_mu > 0.05) & (data[0] > 0.5)
    return ~condition


flat_lcdm_params = (67.7, 0.31, 0.69)

model = model_maker.basic_model

H0_range = np.linspace(67, 79, 21)
omega_m_range = np.linspace(0.30, 0.35, 21)
omega_lambda_range = np.linspace(0.7, 0.65, 21)
combinations = np.array(np.meshgrid(H0_range, omega_m_range, omega_lambda_range)).T.reshape(-1, 3)

descriptions = [['gaussian'], ['gaussian', 'skew'], ['gaussian', 'skew', 'malmquist']]
outs = []

for description in descriptions:
    data, sigma_data = get_data(model, flat_lcdm_params, description)
    best_fit_params = chi2_fit_all(combinations, data, sigma_data, model)
    outs.append("{} : {}".format(description, best_fit_params))

for i in outs:
    print(i)




