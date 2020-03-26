# This generates mock redshifts and corresponding magnitudes based on the model
# get output by calling get_data() which gives np array with a[0] as z and a[1] as mags

import numpy as np
import model_maker


def get_data():
    n = 100
    model = model_maker.get_model()

    # generate SN and apply model
    z = np.random.random_sample([n])
    mags = model(z)

    # adding noise
    sigma = 0.01  # mag
    noise = np.random.normal(0, sigma, n)
    mags += noise

    data = np.array([z, mags])
    return data
