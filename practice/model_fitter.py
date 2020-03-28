import matplotlib.pyplot as plt
import numpy as np

import model_maker
import data_maker
import hubble_diagram

flat_lcdm_params = (67.7, 0.31, 0.69)

model = model_maker.basic_model
data, sigma_data = data_maker.get_data(model, flat_lcdm_params)

H0_range = np.linspace(65, 70, 10)
omega_m_range = np.linspace(0.25, 0.35, 10)
omega_lambda_range = np.linspace(0.65, 0.75, 10)

combinations = np.array(np.meshgrid(H0_range, omega_m_range, omega_lambda_range)).T.reshape(-1, 3)

sn_z = data[0]
true_mags = np.array([model(i, flat_lcdm_params) for i in sn_z])
sigma_mags = sigma_data[1]

all_chi2 = np.empty(np.shape(combinations)[0])
print("{} calcs to do".format(np.shape(combinations)[0]))
for i in range(np.shape(combinations)[0]):
    if i % 1000 == 0:
        print(i)
    model_params = tuple(combinations[i])
    model_mags = np.array([model(i, model_params) for i in sn_z])
    chi2 = np.sum(((true_mags - model_mags) / sigma_mags)**2)
    all_chi2[i] = chi2

best_fit_idx = np.argmin(all_chi2)
best_fit_params = tuple(combinations[best_fit_idx])
print("best fit: {}".format(best_fit_params))

fig, ax = plt.subplots()

hubble_diagram.plot_data(ax, data, sigma_data[1])
hubble_diagram.plot_model(ax, model, best_fit_params, linestyle=':', c='black')
hubble_diagram.plot_model(ax, model, flat_lcdm_params, linestyle=':', c='blue')

# decoration
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.xscale("log")
plt.xlim(0.09, 1)
plt.ylim(36, 45)
plt.show()

# difference between best fit and lcdm
fig, ax = plt.subplots()

hubble_diagram.plot_difference(ax, model, flat_lcdm_params, flat_lcdm_params, linestyle=':', c='blue')
hubble_diagram.plot_difference(ax, model, flat_lcdm_params, best_fit_params, linestyle=':', c='black')

plt.xlabel("z")
plt.ylabel("$\mu - \mu_{\Lambda cdm}$")
plt.xscale("log")
plt.xlim(0.09, 1)
plt.ylim(-0.05, 0.05)
plt.show()


likelihood = np.exp(-all_chi2/2)



