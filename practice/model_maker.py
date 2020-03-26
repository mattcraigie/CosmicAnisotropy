# This produces a model based on input cosmology parameters
import numpy as np
from scipy.integrate import quad

c = 299792  # km/s

def basic_model(z, cosmoparams):
    # input cosmoparams as a tuple with (H0, omega_M, omega_Lambda), but might extend to w etc.
    H0, omega_m, omega_lambda = cosmoparams
    return distance_modulus(luminosity_distance(z, cosmoparams))


def distance_modulus(luminosity_distance):
    print(luminosity_distance)
    return 5 * np.log10(luminosity_distance / 10.)


def hubble_parameter(zprime, cosmoparams):
    H0, omega_m, omega_lambda = cosmoparams
    w = -1.0
    return H0 * np.sqrt(omega_m * (1+zprime)**3 + omega_lambda * (1+zprime)**(3*(1+w)))


def luminosity_distance(z, cosmoparams):
    # quad likes to unpack when you supply args so they're in a list to input
    integral, _ = quad(lambda x: 1 / hubble_parameter(x, cosmoparams), 0., z)
    lum_dist_mpc = (1+z) * c * integral
    return lum_dist_mpc




