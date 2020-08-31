import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatwCDM, wCDM
from scipy.stats import skewnorm

z = np.linspace(0.01, 1, 1000)

H0 = 70
om = 0.3
w = -1


fiducial = FlatwCDM(H0=H0, Om0=om, w0=w).distmod(z).value

# create a randomly scattered SN sample
sn_n = 1000
sn_z = np.random.random_sample(sn_n)
sn_mu_gaussian = np.random.normal(loc=0, scale=0.1, size=sn_n)

# adding lensing skewness
rand_skew = skewnorm.rvs(a=10, size=sn_n)
sn_mu_lensed = sn_z * 0.1 * (rand_skew - np.mean(skewnorm.rvs(a=10, size=10000)))


# ---plotting---
fig, ax = plt.subplots()

# fiducial
h, = ax.plot(z, fiducial-fiducial, c='black', linestyle='--', linewidth=2)
h.set_label('$\Omega_m=0.3, w=-1$')

# supernovae
ax.scatter(sn_z, sn_mu_gaussian + sn_mu_lensed, s=2, c='royalblue')


plt.legend()
plt.ylim([-0.5, 0.5])
plt.show()

