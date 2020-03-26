# This produces a model based on input cosmology parameters
import numpy as np

def basic_model(z):
    return np.sin(4*np.pi*z)


def get_model():
    return basic_model


