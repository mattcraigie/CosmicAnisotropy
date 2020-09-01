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


def load_biascor_results(fname):
    """Load in the biascor best fit results from the pippin output file, e.g. 'all_biascor.csv'"""
    bcor_res = pd.read_csv(fname, usecols=['name', 'OM', 'OM_sig', 'w', 'wsig_marg'])
    bcor_res['name'] = [n[:n.find('FIELD')] for n in bcor_res['name'].values]
    bcor_res = bcor_res.set_index('name')
    return bcor_res


def get_mcmc_fits(field_name):
    """Get the mcmc fits from inside the gzip files that pippin outputs"""

    c = ChainConsumer()
    df = pd.read_csv('(NOHUBBLE + SN) ALL_' + field_name + 'FIELD_ALL.csv.gz')
    weights, likelihood, chain, labels = df["_weight"].values, df["_likelihood"].values, df.iloc[:, 2:].to_numpy(), list(df.columns[2:])

    c.add_chain(chain, weights=weights, parameters=labels, name=field_name, posterior=likelihood)
    summary = c.analysis.get_summary()

    return summary[list(summary.keys())[0]][1], summary[list(summary.keys())[1]][1]


def get_best_fits(field_names):
    """Get best fits for each field and return them in a pandas df"""
    best_fit_dict = {'field': [], 'om': [], 'w': []}

    for field in field_names:
        best_fit_dict['field'].append(field)
        om, w = get_mcmc_fits(field)
        best_fit_dict['om'].append(om)
        best_fit_dict['w'].append(w)

    df = pd.DataFrame.from_dict(best_fit_dict)
    df = df.set_index('field')
    return df


def get_fitres_files(location):
    """Find the fitres files in the current directory"""
    dirs = []
    for file in os.listdir(location):
        if file.endswith(".fitres"):
            dirs.append(os.path.join(location, file))
    return dirs


def apply_cc_cut(df):
    """Applying the CC contamination cutoff, need to check what value is appropriate"""
    return df[df['PROBCC_BEAMS'] < 0.01]


def get_hubble_residuals(mu, z, model):
    """Determines the hubble residuals for the 'all' fit"""
    return mu - model(z)


def get_adjusted_mu_shift(z_values, mu_values, muerr_values, om, w):
    """Using a manual chisq fit to correct the vshift, since there is no value specified. I arbitrarily pick H0=70 and
    correct from there."""
    mu_shifted = lambda z0, shift: FlatwCDM(H0=70, Om0=om, w0=w).distmod(z0).value + shift
    popt, pcov = curve_fit(mu_shifted, z_values, mu_values, sigma=muerr_values)
    [corrected_shift] = popt
    return corrected_shift


def fix_lowz(df):
    """Fixing the field name of low z sample from a nan to LOWZ"""
    mask = pd.isna(df['FIELD'])
    df['FIELD'] = df['FIELD'].mask(mask, other='LOWZ')
    return df


def drop_lowz(df):
    """Removes the LOWZ sample which shows up as na in the field column"""
    df = df.dropna(subset=['FIELD'])
    return df


def make_hubble_residual_plot(fitres_files, best_fits):

    om = best_fits.loc['ALL']['om']
    w = best_fits.loc['ALL']['w']
    df = pd.read_csv(fitres_files[0], delim_whitespace=True, comment="#")
    df = fix_lowz(df)
    df = apply_cc_cut(df)

    z = df["zHD"]
    mu = df["MU"]
    muerr = df['MUERR']

    mu_shift = get_adjusted_mu_shift(z, mu, muerr, om, w)
    all_model = lambda redshift: FlatwCDM(H0=70, Om0=om, w0=w).distmod(redshift).value + mu_shift

    residuals = get_hubble_residuals(mu, z, all_model)
    df['RESIDUALS'] = residuals

    fig, ax = plt.subplots(figsize=(8, 6))

    for field in df.groupby(by='FIELD'):
        field_name = field[0][0] if field[0][0] != 'L' else field[0][:4]
        ax.scatter(field[1]['zHD'], field[1]['RESIDUALS'], c=cdict[field_name], s=10)

    plot_best_fits(ax, best_fits, fitres_files, all_model)

    plt.show()
    return df


def plot_best_fits(ax, best_fits, fitres_files, all_model):
    print(best_fits)
    z_lin = np.linspace(0.01, 1.1, 1000)

    ax.plot(z_lin, all_model(z_lin) - all_model(z_lin), c='black')

    for field in field_names[1:]:

        om = best_fits.loc[field]['om']
        w = best_fits.loc[field]['w']
        df = pd.read_csv(field + fitres_files[0][5:], delim_whitespace=True, comment="#")
        df = apply_cc_cut(df)

        z = df["zHD"]
        mu = df["MU"]
        muerr = df['MUERR']

        # mu = all_model(z)
        muerr = z * 0 + 0.01

        mu_shift = get_adjusted_mu_shift(z, mu, muerr, om, w)
        field_model = lambda redshift: FlatwCDM(H0=70, Om0=om, w0=w).distmod(redshift).value + mu_shift

        residual_best_fit = field_model(z_lin) - all_model(z_lin)
        ax.plot(z_lin, residual_best_fit, c=cdict[field])


if __name__ == "__main__":
    field_names = ['ALL', 'C', 'E', 'S', 'X']
    cdict = {'ALL': 'black', 'C': 'salmon', 'E': 'deepskyblue', 'S': 'goldenrod', 'X': 'purple', 'LOWZ': 'grey'}



    best_fits = get_best_fits(field_names)
    fitres_files = get_fitres_files(".")

    make_hubble_residual_plot(fitres_files, best_fits)

