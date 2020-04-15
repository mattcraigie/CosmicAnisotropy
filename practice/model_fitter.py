import matplotlib.pyplot as plt
import numpy as np

import model_maker
import data_maker_extended
import hubble_diagram


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


flat_lcdm_params = (67.7, 0.31, 0.69)

model = model_maker.basic_model
# data, sigma_data = data_maker.get_data(model, flat_lcdm_params)
# pre_cut_data, pre_cut_sigma, data, sigma_data = data_maker.get_data_malmquist(model, flat_lcdm_params)

data, sigma_data, data_cut, sigma_data_cut = data_maker_extended.get_data(model, flat_lcdm_params)

H0_range = np.linspace(67, 79, 10)
omega_m_range = np.linspace(0.30, 0.35, 10)
omega_lambda_range = np.linspace(0.7, 0.65, 10)

combinations = np.array(np.meshgrid(H0_range, omega_m_range, omega_lambda_range)).T.reshape(-1, 3)

## EDIT THESE IF USING MALMQUIST
best_fit_params = chi2_fit_all(combinations, data_cut, sigma_data_cut, model)
print(best_fit_params)

# fig, ax = plt.subplots()
#
# hubble_diagram.plot_data(ax, data, sigma_data[1])
# hubble_diagram.plot_data(ax, data_cut, sigma_data_cut[1])
#
# hubble_diagram.plot_model(ax, model, best_fit_params, linestyle='-', c='black')
# hubble_diagram.plot_model(ax, model, flat_lcdm_params, linestyle='--', c='blue')
#
# # alternative fixed models
# params_1 = (67.7, 1, 0)
# params_2 = (67.7, 0.3, 0)
#
# hubble_diagram.plot_model(ax, model, params_1, linestyle='--', c='green', alpha=0.5)
# hubble_diagram.plot_model(ax, model, params_2, linestyle='--', c='orange', alpha=0.5)

# alternative marginalised models

# ### MODEL 1 ###
# n_to_marginalise = 200
# H0_range_to_marginalise = np.linspace(50, 70, n_to_marginalise)
# omega_m_fixed = np.ones(n_to_marginalise) * 0.3
# omega_lambda_fixed = np.ones(n_to_marginalise) * 0.0
#
# combinations = np.array([H0_range_to_marginalise, omega_m_fixed, omega_lambda_fixed]).T.reshape(-1, 3)
# print(combinations)
# marginalised_1 = chi2_fit_all(combinations, data, sigma_data, model)
# print()
#
#
# ### MODEL 2 ###
# n_to_marginalise = 200
# H0_range_to_marginalise = np.linspace(50, 70, n_to_marginalise)
# omega_m_fixed = np.ones(n_to_marginalise) * 1
# omega_lambda_fixed = np.ones(n_to_marginalise) * 0.0
#
# combinations = np.array([H0_range_to_marginalise, omega_m_fixed, omega_lambda_fixed]).T.reshape(-1, 3)
# marginalised_2 = chi2_fit_all(combinations, data, sigma_data, model)
#

# hubble_diagram.plot_model(ax, model, marginalised_1, linestyle=':', c='green', alpha=0.5)
# hubble_diagram.plot_model(ax, model, marginalised_2, linestyle=':', c='orange', alpha=0.5)


#
# plt.title("Hubble diagram with mock points")
# # decoration
# plt.xlabel("z")
# plt.ylabel("$\mu$")
# plt.xscale("log")
# plt.xlim(0.09, 1)
# plt.ylim(36, 45)
# plt.show()

# difference between best fit and lcdm
fig, ax = plt.subplots()

# best fit (i.e. flat line)
hubble_diagram.plot_difference(ax, model, best_fit_params, best_fit_params,  linestyle='-', c='black', linewidth=2)

# lambda CDM
hubble_diagram.plot_difference(ax, model, best_fit_params, flat_lcdm_params,  linestyle='--', c='deepskyblue', linewidth=2)

# fixed params
# hubble_diagram.plot_difference(ax, model, best_fit_params, params_1,  linestyle='--', c='green', alpha=0.5)
# hubble_diagram.plot_difference(ax, model, best_fit_params, params_2,  linestyle='--', c='orange', alpha=0.5)

# marginalised params
# hubble_diagram.plot_difference(ax, model, best_fit_params, marginalised_1,  linestyle=':', c='orange', linewidth=2)
# hubble_diagram.plot_difference(ax, model, best_fit_params, marginalised_2,  linestyle=':', c='green', linewidth=2)

# Data
hubble_diagram.plot_difference_data(ax, model, best_fit_params, data, sigma_data[1], c='blue')
hubble_diagram.plot_difference_data(ax, model, best_fit_params, data_cut, sigma_data_cut[1], c='red')


plt.title("Residuals from best fit")
plt.xlabel("z")
plt.ylabel("$\mu - \mu_{best fit}$")
plt.xscale("log")
plt.xlim(0.09, 1)
plt.ylim(-1, 1)
plt.show()




