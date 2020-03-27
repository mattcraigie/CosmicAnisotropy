# This is the top level program that takes a model and SN data and plots
import matplotlib.pyplot as plt
import numpy as np

import model_maker
import data_maker


def plot_data(ax, data):
    # can either be logged or unlogged
    data_z = data[0]
    data_mag = data[1]

    ax.scatter(data_z, data_mag, s=10, alpha=0.5)
    return


def plot_model(ax, model, cosmoparams, **kwargs):
    zrange = np.linspace(0.01, 1, 200)
    mags = np.array([model(i, cosmoparams) for i in zrange])
    ax.plot(zrange, mags, **kwargs)
    return


def main_plot():
    # universe models: (H0, omega_m, omega_lambda)
    flat_lcdm_params = (67.7, 0.31, 0.69)
    open_matter_params = (67.7, 1.0, 1.0)  # note that given these universes we would find a different value of H_0...
    flat_matter_params = (67.7, 1.0, 0.0)

    model = model_maker.basic_model

    data = data_maker.get_data(model, flat_lcdm_params)

    fig, ax = plt.subplots()

    plot_data(ax, data)
    plot_model(ax, model, flat_lcdm_params, linestyle=':', c='black')

    plot_model(ax, model, open_matter_params, linestyle=':', c='blue')
    plot_model(ax, model, flat_matter_params, linestyle=':', c='red')

    # decoration
    plt.xlabel("z")
    plt.ylabel("$\mu$")
    plt.xscale("log")
    plt.xlim(0.01, 1)
    plt.ylim(32.5, 45)

    plt.show()


main_plot()