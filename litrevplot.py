import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

names = ['SNe Ia (Cepheids)', 'SNe Ia (TRGB Freedman)', 'SNe Ia (TRGB Yuan)', 'TF Relation', 'SBF', 'GW', 'CMB', 'BAO + CMB', 'BAO + BBN', 'BAO + CMB + SNe Ia', 'GLTD', 'GLTD + SNe Ia']
H0 = [74.03, 69.8, 72.4, 76.0, 72.0, 68.0, 67.36, 67.6, 67.3, 68.42, 73.3, 73.6]
err = [1.42, 1.9, 2.0, 3.4, 12, 7, 0.54, 0.5, 1.2, 0.88, 1.8, 1.8]

# put in GW error separately

fig, ax = plt.subplots()

for i, name, H0, err in zip(reversed(range(len(names))), names, H0, err):
    local = [0, 1, 6, 7, 8, 9, 10, 11, 12]
    if i in local:
        colour = 'tomato'
    else:
        colour = 'deepskyblue'


    if i < 6:
        i += -0.5
    else:
        i += 0.5

    ax.scatter(H0, i, c=colour)
    if name == 'GW':
        ax.errorbar(H0, i, xerr=np.array([[7], [14]]), c=colour)
    elif name == 'BAO + BBN':
        ax.errorbar(H0, i, xerr=np.array([[1.2], [1.1]]), c=colour)
    elif name == 'GLTD':
        ax.errorbar(H0, i, xerr=np.array([[1.7], [1.8]]), c=colour)
    elif name == 'GLTD + SNe Ia':
        ax.errorbar(H0, i, xerr=np.array([[1.8], [1.6]]), c=colour)
    else:
        ax.errorbar(H0, i, xerr=err, c=colour)



    plt.text(H0, i+0.3, name, horizontalalignment='center', verticalalignment = 'center', fontsize=10)


ax.yaxis.set_visible(False)
plt.title("$H_0$ as Measured by Various Probes", fontsize=14)
plt.xlim(55, 85)
plt.ylim(-1.5, 12.5)
plt.xlabel("$H_0$ (km s$^{-1}$ Mpc$^{-1}$)", fontsize=14)

ax.scatter([-10], [-10], c=['tomato'], label='local')
ax.scatter([-10], [-10], c=['deepskyblue'], label='distant')

plt.legend(loc='lower left', fontsize=12)

plt.axvline(67.36, c='deepskyblue', linestyle=':', alpha=0.7, zorder=-1)
plt.axvline(74.03, c='tomato', linestyle=':', alpha=0.7, zorder=-1)
plt.axhline(5.5, c='black', linestyle='--', alpha=0.8)

plt.text(55.7, 11.7, 'Candles', fontsize=15)
plt.text(55.7, 4.7, 'Rulers', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)

plt.show()
