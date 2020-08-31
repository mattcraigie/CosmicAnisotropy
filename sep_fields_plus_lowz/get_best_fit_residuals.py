import numpy as np
import yaml
from chainconsumer import ChainConsumer
import pandas as pd
import sys
import argparse
import os
import logging
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from astropy.cosmology import FlatwCDM
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, minimize


def make_hubble_plot_combined(fitres_files, m0diff_file, prob_col_names, fit_params):
    blinding = True

    # changed from concise plotting to easy to understand plotting
    logging.info("Making Hubble plot")
    cdict = {'X': 'red', 'C': 'violet', 'S': 'gold', 'E': 'navy'}

    # plotting
    fig, axes = plt.subplots(figsize=(7, 5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0})

    main_ax = axes[0]
    resid_ax = axes[1]

    # residual axis settings
    resid_ax.set_ylabel(r"$\Delta \mu$")
    resid_ax.tick_params(top=True, which="both")
    resid_ax.set_ylim(-0.5, 0.5)
    resid_ax.set_xlabel("$z$")

    # main axis settings
    main_ax.set_ylabel(r"$\mu$")
    main_ax.set_xlim(0.0, 1.1)
    main_ax.set_ylim(38.5, 45)

    # Plot ref cosmology
    # main_ax.plot(zs, all_distmod, c="k", zorder=-1, lw=0.5, alpha=0.7)
    # resid_ax.plot(zs, all_distmod - all_distmod, c="k", zorder=-1, lw=0.5, alpha=0.7)

    # Loading data and plotting
    for fitres_file, prob_col in zip(fitres_files, prob_cols):
        print(fitres_file)
        field_name, sim_num, *_ = fitres_file.split("_")

        field_letter = field_name.replace('FIELD', '')

        if field_letter == 'ALL':
            # we don't want to plot all the points, but we do want to fit our cosmology
            df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")
            z = df["zHD"]
            mu = df["MU"]
            muerr = df['MUERR']
            om, w = fit_params['ALL']

            zs = np.linspace(0.01, 1.1, 500)

            # method 1: using scipy curve fit
            # mu_shifted = lambda z0, shift: FlatwCDM(H0=70, Om0=om, w0=w).distmod(z0).value + shift
            # popt, pcov = curve_fit(mu_shifted, z, mu, sigma=muerr)
            # corrected_shift = popt
            # all_distmod = mu_shifted(zs, corrected_shift)

            # method 2: using a manual chisq fit
            mu_shifted = lambda x, zs: FlatwCDM(H0=70, Om0=om, w0=w).distmod(zs).value + x

            def bestfitchisq(shift, z, mu, muerr):
                model = mu_shifted(shift, z)
                chisq = np.sum((mu - model) ** 2 / (muerr) ** 2)
                return chisq

            res = minimize(bestfitchisq, x0=np.array([0.1]), args=(z, mu, muerr))
            corrected_shift = res['x'][0]
            all_distmod = mu_shifted(corrected_shift, zs)






            main_ax.plot(zs, all_distmod, c="k", zorder=-1, lw=0.5, alpha=0.7)
            resid_ax.plot(zs, all_distmod - all_distmod, c="k", zorder=-1, lw=2, alpha=0.7)
            continue


        df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")
        dfm = pd.read_csv(m0diff_file)
        # data cuts
        # dfm = dfm[(dfm.name == name) & (dfm.sim_num == sim_num) & (dfm.muopt_num == 0) & (dfm.fitopt_num == 0)]
        df.sort_values(by="zHD", inplace=True)
        # dfm.sort_values(by="z", inplace=True)
        dfm = dfm[dfm["MUDIFERR"] < 10]

        dfz = df["zHD"]

        # change alpha based on prob Ia. Cannot input an alpha array so must set rgba colour.
        alphamap = df[prob_col]

        # for main scatter
        rgb = mcolors.to_rgb(cdict[field_letter])  # get this field's colour
        rgba = np.zeros((len(alphamap), 4))
        for i in [0, 1, 2]:
            rgba[:, i] = rgb[i]
        rgba_main = rgba.copy()
        rgba_main[:, 3] = alphamap

        # for the residuals make a tint so it looks a little nicer
        rgba_resid = rgba.copy() + (1 - rgba.copy()) * 0.5
        rgba_resid[:, 3] = alphamap * 0.5

        # for errorbar (yeah I shouldn't do it four times but I'm not taking it out of the loop that's ugly)
        rgb = mcolors.to_rgb("#AAAAAA")  # AAAAAAAAAaaaA
        rgba_errbar = np.zeros((len(alphamap), 4))
        for i in [0, 1, 2]:
            rgba_errbar[:, i] = rgb[i]
        rgba_errbar[:, 3] = alphamap * 0.4

        # main axis plotting
        main_ax.errorbar(dfz, df["MU"], yerr=df["MUERR"], fmt="none", elinewidth=0.5, c=rgba_errbar)
        main_ax.scatter(dfz, df["MU"], c=rgba_main, s=2, zorder=2)

        # residual axis plotting
        resid_ax.errorbar(dfz, df["MU"] - df["MUMODEL"], yerr=df["MUERR"], fmt="none", elinewidth=0.5, c=rgba_errbar)
        resid_ax.scatter(dfz, df["MU"] - df["MUMODEL"], c=rgba_resid, s=2, zorder=2)

        # field best fit
        om, w = fit_params[field_letter]

        # curve fit
        # mu_shifted = lambda z0, shift: FlatwCDM(H0=70, Om0=om, w0=w).distmod(z0).value + shift
        # popt, pcov = curve_fit(mu_shifted, dfz, df["MU"], sigma=df["MUERR"])
        # corrected_shift = popt

        # using minimise and manually minimising the chisq
        mu_shifted = lambda x, zs: FlatwCDM(H0=70, Om0=om, w0=w).distmod(zs).value + x

        def bestfitchisq(shift, z, mu, muerr):
            model = mu_shifted(shift, z)
            chisq = np.sum((mu - model)**2 / (muerr)**2)
            return chisq

        res = minimize(bestfitchisq, x0=np.array([0.1]), args=(dfz, df["MU"], df["MUERR"]))
        corrected_shift = res['x'][0]

        zs = np.linspace(0.01, 1.1, 500)
        distmod = mu_shifted(corrected_shift, zs)

        # field cosmology plotting
        main_ax.plot(zs, distmod, c=cdict[field_letter], linestyle=':', zorder=3, lw=0.5, alpha=1)
        resid_ax.plot(zs, distmod - all_distmod, c=cdict[field_letter], linestyle=':', zorder=3, lw=2, alpha=1)

    handles = [mpatches.Patch(color=cdict[l], label=l + ' Region') for l in ['X', 'S', 'E', 'C']]
    main_ax.legend(handles=handles, loc='lower right')
    plt.show()
    fp = "hubble_combined.png"
    logging.debug(f"Saving combined Hubble plot to {fp}")
    fig.savefig(fp, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)

    # Todo: Fix the vertical shift
    # Todo: Put lowz SNe on the plot somehow -- also I need to include them in the fits or its all wrong
    # Where are they??

    return


def load_biascor(fname):
    bcor_res = pd.read_csv(fname, usecols=['name', 'OM', 'OM_sig', 'w', 'wsig_marg'])
    bcor_res['name'] = [n[:n.find('FIELD')] for n in bcor_res['name'].values]
    bcor_res = bcor_res.set_index('name')
    return bcor_res


def get_fitres_files(location):
    dirs = []
    for file in os.listdir(location):
        if file.endswith(".fitres"):
            dirs.append(os.path.join(location, file))
    return dirs


def get_hubble_residuals(best_fits, fitres_files):
    om = best_fits.loc['ALL']['OM']
    w = best_fits.loc['ALL']['w']

    df = pd.read_csv(fitres_files[0], delim_whitespace=True, comment="#")

    # apply CC cut
    df = df[df['PROBCC_BEAMS'] < 0.1]

    z = df["zHD"]
    mu = df["MU"]
    muerr = df['MUERR']

    mu, resid = correct_vshift(z, mu, muerr, om, w)

    return


def correct_vshift(z_values, mu_values, muerr_values, om, w):
    """Using a manual chisq fit to correct the vshift, since there is no value specified. I arbitrarily pick H0=70 and
    correct from there."""


    model_shifted = lambda x, z: FlatwCDM(H0=70, Om0=om, w0=w).distmod(z).value + x
    fit_chisq = lambda x, z, mu, muerr: np.sum((mu - model_shifted(x, z)) ** 2 / (muerr) ** 2)

    res = minimize(fit_chisq, x0=np.array([0.1]), args=(z_values, mu_values, muerr_values))
    corrected_shift = res['x'][0]
    print(corrected_shift)
    mu_values_corrected = mu_values - corrected_shift

    print(mu_values.values)
    print(mu_values_corrected.values)

    residuals = mu_values_corrected - mu_shifted(corrected_shift, z_values)
    print(np.average(residuals, weights=1/muerr_values**2))

    plt.axhline(0)
    plt.scatter(z_values, residuals, s=1)
    plt.show()

    plt.hist(residuals)
    plt.show()


    return mu_values_corrected, residuals


def plot_or_something(best_fits, fitres_files):
    df = pd.read_csv(fitres_files, delim_whitespace=True, comment="#")
    field_name = fitres_files[fitres_files.find('\\') + 1:fitres_files.find('FIELD')]

    print(best_fits)

    om = best_fits.loc[field_name]['OM']
    w = best_fits.loc[field_name]['w']


if __name__ == "__main__":


    # how I want this to run
    best_fits = load_biascor('all_biascor.csv')
    fitres_files = get_fitres_files(".")

    hr = get_hubble_residuals(best_fits, fitres_files)

    exit()
    fit_params = {}

    for i, pb in enumerate(param_bounds):
        try:
            fit_params[fields[i]] = (pb['$\\Omega_m\\ \\mathrm{Blinded}$'][1], pb['$w\\ \\mathrm{Blinded}$'][1])
        except KeyError:
            fit_params[fields[i]] = (pb['$\\Omega_m$'][1], pb['$w$'][1])



    m0diff_file = args2.get("M0DIFF_PARSED")
    fitres_files = args2.get("FITRES_PARSED")
    prob_cols = args2.get("FITRES_PROB_COLS")
    # for f, p in zip(fitres_files, prob_cols):
    # make_hubble_plot_simple(f, m0diff_file, best_cosmology=(omega_m_best, w_best))
    make_hubble_plot_combined(fitres_files, m0diff_file, prob_cols, fit_params)

