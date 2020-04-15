# This produces a model based on input cosmology parameters
import numpy as np
from scipy.integrate import quad

c = 299792  # km/s


def basic_model(z, cosmoparams):
    # input cosmoparams as a tuple with (H0, omega_M, omega_Lambda), but might extend to w etc.
    H0, omega_m, omega_lambda = cosmoparams
    return distance_modulus(luminosity_distance(z, cosmoparams))


def distance_modulus(luminosity_distance):
    return 5 * np.log10(luminosity_distance / 10.)


def hubble_parameter(zprime, cosmoparams):
    H0, omega_m, omega_lambda = cosmoparams
    omega_k = 1 - omega_m - omega_lambda
    w = -0.978
    return H0 * np.sqrt(omega_m * (1+zprime)**3 + omega_lambda * (1+zprime)**(3*(1+w)) + omega_k * (1+zprime)**2)


def luminosity_distance(z, cosmoparams):
    # quad likes to unpack when you supply args so they're in a list to input
    integral, _ = quad(lambda x: 1 / hubble_parameter(x, cosmoparams), 0., z)
    comoving_distance = c * integral

    H0, omega_m, omega_lambda = cosmoparams
    omega_k = 1 - omega_m - omega_lambda

    if omega_k > 0:  # hyperbolic
        d_M = c / H0 * np.sinh(np.sqrt(np.abs(omega_k)) * comoving_distance * H0 / c) / np.sqrt(np.abs(omega_k))
    elif omega_k < 0:  # spherical
        d_M = c / H0 * np.sin(np.sqrt(np.abs(omega_k)) * comoving_distance * H0 / c) / np.sqrt(np.abs(omega_k))
    else:
        d_M = comoving_distance

    lum_dist_mpc = (1+z) * d_M
    return lum_dist_mpc * 1000000




