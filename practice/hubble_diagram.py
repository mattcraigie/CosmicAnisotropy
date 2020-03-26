# This is the top level program that takes a model and SN data and plots
import matplotlib.pyplot as plt
import numpy as np

import model_maker
import data_maker


def plot_data(ax, data):
    # can either be logged or unlogged
    data_z = data[0]
    data_mag = data[1]

    ax.scatter(data_z, data_mag)
    return


def plot_model(ax, model, cosmoparams):
    zrange = np.linspace(0.01, 1, 200)
    mags = np.array([model(i, cosmoparams) for i in zrange])
    ax.plot(zrange, mags)
    return


def main_plot():
    cosmoparams = 70, 0.2, 0.8
    model = model_maker.basic_model

    data = data_maker.get_data(model, cosmoparams)

    fig, ax = plt.subplots()

    plot_data(ax, data)
    plot_model(ax, model, cosmoparams)

    # decoration
    plt.xlabel("z")
    plt.ylabel("$\mu$")
    plt.xscale("log")
    plt.xlim(0.01, 1)

    plt.show()


main_plot()