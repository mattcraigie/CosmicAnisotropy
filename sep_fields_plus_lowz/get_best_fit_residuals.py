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
from scipy.stats import skewtest
import colorsys


def load_biascor_results(fname):
    """
    *** TURNS OUT THIS ISN'T RIGHT***
    Load in the biascor best fit results from the pippin output file, e.g. 'all_biascor.csv'"""
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


def get_fitres_files(location, fields):
    """Find the fitres files in the current directory"""
    dirs = []
    for file in os.listdir(location):
        if file.endswith(".fitres"):
            dirs.append(os.path.join(location, file))

    print(dirs)

    file_dict = {}
    for field in fields:

        for dir_ in dirs:
            if dir_[2:].startswith(field):
                file_dict[field] = dir_

    return file_dict


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


def get_data_from_fitres(fitres_files, field, cc_cut=True, remove_lowz=False):
    df = pd.read_csv(fitres_files[field], delim_whitespace=True, comment="#")

    if remove_lowz:
        df = drop_lowz(df)
    else:
        df = fix_lowz(df)

    if cc_cut:
        df = apply_cc_cut(df)

    df['z'] = df['zHD']
    df['mu'] = df['MU']
    df['muerr'] = df['MUERR']
    df['field'] = [i[0] if i[0] != 'L' else i for i in df['FIELD']]
    return df


def make_hubble_residual_plot(fitres_files, best_fits):

    om = best_fits.loc['ALL']['om']
    w = best_fits.loc['ALL']['w']

    all_df = get_data_from_fitres(fitres_files, 'ALL', cc_cut=True, remove_lowz=False)

    mu_shift = get_adjusted_mu_shift(all_df['z'], all_df['mu'], all_df['muerr'], om, w)
    all_model = lambda redshift: FlatwCDM(H0=70, Om0=om, w0=w).distmod(redshift).value + mu_shift

    residuals = get_hubble_residuals(all_df['mu'], all_df['z'], all_model)
    all_df['resid'] = residuals

    fig, ax = plt.subplots(figsize=(8, 6))

    for field in all_df.groupby(by='field'):
        field_name = field[0][0] if field[0][0] != 'L' else field[0][:4]
        ax.scatter(field[1]['z'], field[1]['resid'], c=cdict[field_name], s=10)

    plot_best_fits(ax, best_fits, fitres_files, all_model)

    plt.show()
    return


def plot_best_fits(ax, best_fits, fitres_files, all_model):
    print(best_fits)
    z_lin = np.linspace(0.01, 1.1, 1000)

    ax.plot(z_lin, all_model(z_lin) - all_model(z_lin), c='black')

    for field in field_names[1:]:

        om = best_fits.loc[field]['om']
        w = best_fits.loc[field]['w']
        field_df = get_data_from_fitres(fitres_files, field, cc_cut=True)

        mu_shift = get_adjusted_mu_shift(field_df['z'], field_df['mu'], field_df['muerr'], om, w)
        field_model = lambda redshift: FlatwCDM(H0=70, Om0=om, w0=w).distmod(redshift).value + mu_shift

        residual_best_fit = field_model(z_lin) - all_model(z_lin)
        ax.plot(z_lin, residual_best_fit, c=cdict[field])


def plot_residual_histograms(fitres_files, best_fits, by='field', bins=20, sharex=False, zbins=5):

    om = best_fits.loc['ALL']['om']
    w = best_fits.loc['ALL']['w']
    all_df = get_data_from_fitres(fitres_files, 'ALL', cc_cut=True, remove_lowz=True)

    mu_shift = get_adjusted_mu_shift(all_df['z'], all_df['mu'], all_df['muerr'], om, w)
    all_model = lambda redshift: FlatwCDM(H0=70, Om0=om, w0=w).distmod(redshift).value + mu_shift
    residuals = get_hubble_residuals(all_df['mu'], all_df['z'], all_model)
    all_df['resid'] = residuals

    if by == 'field':
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4), sharex=sharex, sharey=True)
    elif by == 'redshift':
        fig, axes = plt.subplots(nrows=1, ncols=zbins+1, figsize=(20, 4), sharex=sharex, sharey=True)
    else:
        raise ValueError

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.2)
    fig.text(0.52, 0.05, 'Number of SNe Ia in residual bin', size=14,
             horizontalalignment='center', verticalalignment='center')

    bin_max = 0.8
    bin_edges = np.linspace(-bin_max, bin_max, bins + 1)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_middles = (bin_edges + bin_width / 2)[:-1]

    # For all data combined
    all_counts, _ = np.histogram(all_df['resid'], bins=bin_edges)

    axes[0].barh(bin_middles, all_counts, bin_width, color='white', edgecolor='black')
    axes[0].set_title('ALL')
    axes[0].set_ylim(-bin_max, bin_max)
    axes[0].set_ylabel('$\mu$ - $\mu_{model}$', size=14)

    wmean = np.average(all_df['resid'], weights=1 / all_df['muerr'] ** 2)
    skew = skewtest(all_df['resid'])
    axes[0].text(0.95, 0.05, 'n: {}\nmean: {:.2f}\nskew $z$: {:.2f}'.format(len(all_df), np.abs(wmean), skew[0]),
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[0].transAxes)

    if by == 'field':

        all_df = all_df[all_df['z'] > 0.6]

        # For each individual field
        for i, field in enumerate(all_df.groupby(by='field')):
            field_residuals = field[1]['resid']
            field_counts, _ = np.histogram(field_residuals, bins=bin_edges)
            inner_colour = scale_lightness(cdict[field[0]], 0.5, lighten=True)
            edge_colour = scale_lightness(cdict[field[0]], 0.5, lighten=False)
            axes[i + 1].barh(bin_middles, field_counts,  bin_width,  color=inner_colour, edgecolor=edge_colour)
            axes[i + 1].set_title(field[0])

            wmean = np.average(field_residuals, weights=1/field[1]['muerr']**2)
            skew = skewtest(field_residuals)
            axes[i + 1].text(0.95, 0.05, 'n: {}\nmean: {:.2f}\nskew $z$: {:.2f}'.format(len(field[1]), wmean, skew[0]),
                             horizontalalignment='right', verticalalignment='bottom', transform=axes[i + 1].transAxes)

        # For all subplots
        for ax in axes:
            ax.set_ylim(-bin_max, bin_max)
            ax.axhline(0, c='black', linewidth=2, linestyle='--')

    if by == 'redshift':
        # For redshift bins

        # set up redshift bins
        zbin_edges = np.linspace(0.2, 1.0, zbins + 1)
        zbin_middles = (zbin_edges + (zbin_edges[1] - zbin_edges[0]) / 2)[:-1]

        # NOTE I AM REMOVING A BUNCH HERE
        all_df = all_df[(all_df['z'] > np.min(zbin_edges)) & (all_df['z'] < np.max(zbin_edges))]
        nearest_binmid = lambda z: zbin_middles[int(np.argmin(np.abs(z - zbin_middles)))]

        all_df['zbin'] = all_df['z'].apply(func=nearest_binmid)

        for i, zbin in enumerate(all_df.groupby(by='zbin')):
            zbin_residuals = zbin[1]['resid']
            zbin_counts, _ = np.histogram(zbin_residuals, bins=bin_edges)
            axes[i + 1].barh(bin_middles, zbin_counts, bin_width, color='salmon', edgecolor='maroon')
            axes[i + 1].set_title('$z={:.2f}$'.format(zbin[0]))

            wmean = np.average(zbin_residuals, weights=1 / zbin[1]['muerr'] ** 2)
            skew = skewtest(zbin_residuals)
            axes[i + 1].text(0.95, 0.05, 'n: {}\nmean: {:.2f}\nskew $z$: {:.2f}'.format(len(zbin[1]), wmean, skew[0]),
                             horizontalalignment='right', verticalalignment='bottom', transform=axes[i + 1].transAxes)

        # For all subplots
        for ax in axes:
            ax.set_ylim(-bin_max, bin_max)
            ax.axhline(0, c='black', linewidth=2, linestyle='--')

    plt.show()

def scale_lightness(col, factor, lighten=True):
    """Borrowed of stackexchange for scaling lightness/darkness of a base colour"""
    rgb = np.array(mcolors.ColorConverter().to_rgb(col))

    if lighten:
        rgb = rgb + (1 - rgb) * factor
    else:
        rgb = rgb * (1 - factor)

    return tuple(rgb)


if __name__ == "__main__":
    field_names = ['ALL', 'C', 'E', 'S', 'X']
    cdict = {'ALL': 'black', 'C': 'salmon', 'E': 'deepskyblue', 'S': 'goldenrod', 'X': 'purple', 'LOWZ': 'grey'}



    best_fits = get_best_fits(field_names)
    fitres_files = get_fitres_files(".", field_names)
    make_hubble_residual_plot(fitres_files, best_fits)
    plot_residual_histograms(fitres_files, best_fits, by='field', zbins=10)


