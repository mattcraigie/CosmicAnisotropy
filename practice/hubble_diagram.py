# This is the top level program that takes a model and SN data and plots
import matplotlib.pyplot as plt
import numpy as np

import model_maker
import data_maker


def plot_data(ax, data):
    data_z = data[0]
    data_mag = data[1]

    ax.scatter(data_z, data_mag)
    return


def plot_model(ax, model):
    zrange = np.linspace(0, 1, 200)
    mags = model(zrange)

    ax.plot(zrange, mags)
    return


def main_plot():
    data = data_maker.get_data()
    model = model_maker.get_model()

    fig, ax = plt.subplots()

    plot_data(ax, data)
    plot_model(ax, model)

    # decoration
    plt.xlabel("z")
    plt.ylabel("mag")

    plt.show()


main_plot()