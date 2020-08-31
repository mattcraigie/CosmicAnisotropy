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
from scipy.stats import skew

def fail(msg, condition=True):
    if condition:
        logging.error(msg)
        raise ValueError(msg)


def setup_logging():
    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[handler, logging.FileHandler(
        "../ALL_SN/plot_cosmomc.log")])
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def get_arguments():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input yml file", type=str)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config.update(config["COSMOMC"])

    if config.get("NAMES") is not None:
        assert len(config["NAMES"]) == len(config["PARSED_FILES"]), (
            "You should specify one name per base file you pass in." + f" Have {len(config['PARSED_FILES'])} base names and {len(config['NAMES'])} names"
        )
    return config


def get_arguments_2():
    # For some reason this is different for biascor and cosmomc plot scrips but I'm just gonna use both of them separately for now bc lazy
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input yml file", type=str)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        config = yaml.safe_load(f)
    config.update(config["BIASCOR"])

    return config


def load_output(basename):
    if os.path.exists(basename):
        logging.warning(f"Loading in pre-saved CSV file from {basename}")
        df = pd.read_csv(basename)
        return df["_weight"].values, df["_likelihood"].values, df.iloc[:, 2:].to_numpy(), list(df.columns[2:])
    else:
        return None


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

    # Loading data for all fields and plotting
    for fitres_file, prob_col in zip(fitres_files, prob_cols):
        # I plot the lowz in this and establish the best fit for all the data regions combined, which I use later

        field_name, sim_num, *_ = fitres_file.split("_")
        field_letter = field_name.replace('FIELD', '')
        if field_letter != 'ALL':
            continue
        else:
            # loading in
            om, w, = fit_params[field_letter]
            df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")

            # df = df.loc[df['PROBCC_BEAMS'] < 0.1]
            # the probability of CC is different in different regions. So this brings some inconsistency...

            field_names = df['FIELD'].unique()
            regions = []

            for i in list(df['FIELD'].values):
                if i is np.nan:
                    regions.append('LOWZ')
                    continue
                else:
                    regions.append(i[0])

            df['REGION'] = regions
            all_df = df
            a = df[df['REGION'].isin (['X', 'C', 'S', 'E'])]['CID'].values



            print('\n\n')
            for fl in ['X', 'C', 'S', 'E', 'LOWZ']:
                print('n in region ' + fl + ': ' + str(len(df[df['REGION'] == fl])))
            print('total: ' + str(len(df)))
            # take a selection to test
            # dflowz = df[df["zHD"] < 0.08]
            # dfhighz = df[df["zHD"] > 0.08]
            # dfhighz = dfhighz.sample(frac=0.25)
            # df = pd.concat([dflowz, dfhighz])
            # print(len(df))

            # fixing the mu axis shift
            zspace = np.linspace(0.01, 1.1, 500)
            scriptm_correction = shift_cosmology_fit(zspace, df["zHD"], df["MU"], df["MUERR"], om, w)

            # correcting the mu for the plotted model
            mu_all_sn = mu_shifted(scriptm_correction, df['zHD'], om, w)
            mu_all_model = mu_shifted(scriptm_correction, zspace, om, w)

            # binned plotting
            bin_num = 11
            binned_hubble_plot(main_ax, resid_ax, bin_num, df["zHD"], df["MU"], df["MUERR"], 'black', zspace, mu_all_model)

            # plotting lowz
            df['MUMODEL_CORR'] = mu_all_sn
            df = df[df["REGION"] == 'LOWZ']
            z = df['zHD']
            mu = df["MU"]
            mu_err = df["MUERR"]
            mu_sn_model = df['MUMODEL_CORR']

            # main axis plotting
            main_ax.errorbar(z, mu, yerr=mu_err, fmt="none", elinewidth=0.5, c='black', alpha=0.5)
            main_ax.scatter(z, mu, c='black', s=2, zorder=2, alpha=0.5)

            # residual axis plotting
            resid_ax.errorbar(z, mu - mu_sn_model, yerr=mu_err, fmt="none", elinewidth=0.5, c='black', alpha=0.3)
            resid_ax.scatter(z, mu - mu_sn_model, c='black', s=2, zorder=2, alpha=0.3)

            # model plotting
            main_ax.plot(zspace, mu_all_model, c="k", zorder=-1, lw=0.5, alpha=0.7)
            resid_ax.plot(zspace, mu_all_model - mu_all_model, c="k", zorder=-1, lw=2, alpha=0.7)

    total_sne = 0
    total_lowz = []
    b = []
    print("\n\nindividual files:\n")
    for fitres_file, prob_col in zip(fitres_files, prob_cols):
        field_name, sim_num, *_ = fitres_file.split("_")
        field_letter = field_name.replace('FIELD', '')
        if field_letter == 'ALL':
            continue

        col = cdict[field_letter]

        # if field_letter != 'C':
        #     continue

        om, w, = fit_params[field_letter]
        df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")

        # I probably need to do a prob Ia cut...
        # df = df.loc[df['PROBCC_BEAMS'] < 0.1]

        zspace = np.linspace(0.01, 1.1, 500)
        scriptm_correction = shift_cosmology_fit(zspace, df["zHD"], df["MU"], df["MUERR"], om, w)
        mu_model = mu_shifted(scriptm_correction, zspace, om, w)



        # remove lowz
        before_lowzcut = len(df)
        df = df.dropna()
        total_lowz.append(before_lowzcut - len(df))
        total_sne += len(df)

        b += list(df['CID'].values)

        print('n in region ' + field_letter + ': ' + str(len(df)))

        # plotting now
        z = df['zHD']
        mu = df["MU"]
        mu_err = df["MUERR"]
        mu_all_interp = np.interp(z, zspace, mu_all_model)

        # main axis plotting
        main_ax.errorbar(z, mu, yerr=mu_err, fmt="none", elinewidth=0.5, c=col, alpha=0.5)
        main_ax.scatter(z, mu, c=col, s=2, zorder=2, alpha=0.5)

        # residual axis plotting
        resid_ax.errorbar(z, mu - mu_all_interp, yerr=mu_err, fmt="none", elinewidth=0.5, c=col, alpha=0.1)
        resid_ax.scatter(z, mu - mu_all_interp, c=col, s=2, zorder=2, alpha=0.1)

        # field cosmology plotting
        main_ax.plot(zspace, mu_model, c=cdict[field_letter], linestyle=':', zorder=3, lw=0.5, alpha=1)
        resid_ax.plot(zspace, mu_model - mu_all_model, c=cdict[field_letter], linestyle=':', zorder=3, lw=2, alpha=1)

        bin_num = 11
        binned_hubble_plot(main_ax, resid_ax, bin_num, z, mu, mu_err, cdict[field_letter], zspace, mu_all_model)
        binned_skew_plot(main_ax, resid_ax, bin_num, z, mu, mu_err, cdict[field_letter], zspace, mu_all_model)

        # plt.show()

    print('n removed in lowz: ', total_lowz)
    print('total: ', total_sne + total_lowz[0])

    handles = [mpatches.Patch(color=cdict[l], label=l + ' Region') for l in ['X', 'S', 'E', 'C']]
    main_ax.legend(handles=handles, loc='lower right')
    plt.show()
    fp = "hubble_combined.png"
    logging.debug(f"Saving combined Hubble plot to {fp}")
    fig.savefig(fp, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


    b = np.array(b)

    #
    #
    # print(a)
    # print(b)
    #
    # print(len(a))
    # print(len(b))
    #
    # print(len(np.unique(a)))
    # print(len(np.unique(b)))
    # #
    # diff_set = np.setdiff1d(a, b)
    #
    # missing_sne = all_df[all_df['CID'].isin(diff_set)]
    # missing_sne = missing_sne.set_index('CID')
    # missing_sne.to_csv('missing_sne_df.txt')
    #
    # print(set(a) - set(b))
    # print(set(b) - set(a))

    return


def make_residual_histograms(all_fitres, all_fit_params):
    # changed from concise plotting to easy to understand plotting
    num_zbins = 10
    # plotting
    fig, axes = plt.subplots(figsize=(8, 8), nrows=5, ncols=2)

    # axis settings
    for ax in axes.flatten():
        ax.set_ylabel(r"$n$")
        ax.tick_params(top=True, which="both")
        ax.set_xlabel("$\Delta\mu$")
        ax.set_xticks([])

    # loading
    om, w, = all_fit_params
    df = pd.read_csv(all_fitres, delim_whitespace=True, comment="#")

    # cutting out CC SNe
    df = df.loc[df['PROBCC_BEAMS'] < 0.1]

    # removing lowz that have nans for their field
    df = df.dropna()

    # fixing the mu axis shift
    zspace = np.linspace(0.01, 1.1, 500)
    scriptm_correction = shift_cosmology_fit(zspace, df["zHD"], df["MU"], df["MUERR"], om, w)

    # correcting the mu for the plotted model
    mu_all_sn = mu_shifted(scriptm_correction, df['zHD'], om, w)
    mu_all_model = mu_shifted(scriptm_correction, zspace, om, w)

    # binned plotting
    num_resid_hist_bins = 20
    resid_hist_bins = np.linspace(-0.5, 0.5, num_resid_hist_bins)
    binned_residual_hists(axes, num_zbins, df["zHD"], df["MU"], df["MUERR"], zspace, mu_all_model, resid_hist_bins)

    plt.tight_layout()
    plt.show()
    fp = "residual_hists.png"
    fig.savefig(fp, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return


def bestfitchisq(shift, z, mu, muerr, om, w):
    model = mu_shifted(shift, z, om, w)
    chisq = np.sum((mu - model) ** 2 / (muerr) ** 2)
    return chisq


def shift_cosmology_fit(zspace, z, mu, muerr, om, w):
    # make a correction vertically because the cosmomc output does it behind the scenes.
    res = minimize(bestfitchisq, x0=np.array([0.1]), args=(z, mu, muerr, om, w))
    corrected_shift = res['x'][0]
    return corrected_shift


def mu_shifted(x, z, om, w):
    # x is shift, z is z you want returned (i.e. zspace or specific sn z)
    return FlatwCDM(H0=70, Om0=om, w0=w).distmod(z).value + x


def binning(z, mu, mu_err, bin_num):
    minz = 0.1  # anything below z=0.1 is a lowz supernovae
    maxz = 1.1  # The highest we have is below this
    bin_edges = np.linspace(minz, maxz, bin_num + 1)[:-1]
    dz = bin_edges[1] - bin_edges[0]

    df = pd.DataFrame({'z': z, 'mu': mu, 'mu_err': mu_err})
    df['bin_middles'] = [np.nan] * len(df)

    for bottom in bin_edges:
        top = bottom + dz
        df.loc[(bottom < df['z']) &  (df['z'] < top), 'bin_middles'] = bottom + dz/2

    # remove lowz
    df = df.dropna(axis=0)
    return df


def get_zbin_means(binned_df, value_column, value_err_column):
    # gets the bin mean for the value_column column in the binned_df, weighted by the error in value_err_column
    bin_middles = binned_df['bin_middles'].unique()
    means = []
    uncertainties = []

    for bin_mid in bin_middles:
        df_bin = binned_df.loc[binned_df['bin_middles'] == bin_mid]
        weights = 1 / df_bin[value_err_column] ** 2
        bin_mean = np.average(df_bin[value_column], weights=weights.values)
        bin_uncertainty = 1 / len(df_bin[value_err_column]) * np.sqrt(np.sum(df_bin[value_err_column] ** 2))
        means.append(bin_mean)
        uncertainties.append(bin_uncertainty)

    return bin_middles, np.array(means), np.array(uncertainties)


def binned_hubble_plot(main_ax, resid_ax, bin_num, z, mu, mu_err, c, zspace, mu_all_model):

    binned_df = binning(z, mu, mu_err, bin_num)
    binned_df['resid'] = binned_df['mu'] - np.interp(binned_df['z'], zspace, mu_all_model)
    z_bin, resid_bin, resid_err_bin = get_zbin_means(binned_df, value_column='resid', value_err_column='mu_err')
    mu_all_interp = np.interp(z_bin, zspace, mu_all_model)

    # main axis plotting
    main_ax.errorbar(z_bin, resid_bin + mu_all_interp, yerr=resid_err_bin, fmt="none", elinewidth=1, c=c)
    main_ax.scatter(z_bin, resid_bin + mu_all_interp, c=c, s=30, zorder=2)

    # residual axis plotting
    resid_ax.errorbar(z_bin, resid_bin, yerr=resid_err_bin, fmt="none", elinewidth=1, c=c)
    if c == 'black':
        resid_ax.scatter(z_bin, resid_bin, c=c, s=30, zorder=5)
    else:
        resid_ax.scatter(z_bin, resid_bin, c=c, s=30, zorder=2)


def get_zbin_hists(binned_df, value_column, hist_bins):
    bin_middles = np.sort(binned_df['bin_middles'].unique())

    counts = []

    for bin_mid in bin_middles:
        df_bin = binned_df.loc[binned_df['bin_middles'] == bin_mid]
        n, hist_middles = np.histogram(df_bin[value_column], bins=hist_bins)
        counts.append(n)
    return counts


def binned_residual_hists(axes, bin_num, z, mu, mu_err, zspace, mu_all_model, resid_hist_bins):

    binned_df = binning(z, mu, mu_err, bin_num)
    binned_df.sort_values(by='bin_middles')
    binned_df['resid'] = binned_df['mu'] - np.interp(binned_df['z'], zspace, mu_all_model)
    resid_hist_counts = get_zbin_hists(binned_df, value_column='resid', hist_bins=resid_hist_bins)

    binwidth = resid_hist_bins[1] - resid_hist_bins[0]
    resid_hist_bins += binwidth
    resid_hist_bins = resid_hist_bins[:-1]

    ### TEST SKEWNESS ###
    print("\n\nSkewness")
    [print("z = {z:.2f}, skew = {sk:.2f}, n = {num}".format(z=z, sk=skew(n), num=np.sum(n))) for (z, n) in zip(np.sort(binned_df['bin_middles'].unique()), resid_hist_counts)]

    [ax.bar(x=resid_hist_bins, height=cnt, width=binwidth) for ax, cnt in zip(axes.flatten(), resid_hist_counts)]
    [ax.set_title('z = {price:.2f}'.format(price=t)) for (ax, t) in zip(axes.flatten(), np.sort(binned_df['bin_middles'].unique()))]
    return


if __name__ == "__main__":

    setup_logging()
    args2 = get_arguments_2()

    m0diff_file = args2.get("M0DIFF_PARSED")
    fitres_files = args2.get("FITRES_PARSED")
    prob_cols = args2.get("FITRES_PROB_COLS")
    fit_params = {'ALL': (3.9422690E-01, -1.0493690E+00), 'X': (0.41521743369805875, -0.8661256102157622),
                  'S': (0.4858292700626809, -1.210227874888057), 'E': (0.4122255917913083, -0.6660605869482867),
                  'C': (0.305341658598659, -0.5752009246536098)}

    make_residual_histograms(fitres_files[0], fit_params['ALL'])

    exit()


    make_hubble_plot_combined(fitres_files, m0diff_file, prob_cols, fit_params)

    exit()

    setup_logging()
    args = get_arguments()
    args2 = get_arguments_2()
    try:
        if args.get("PARSED_FILES"):
            logging.info("Creating chain consumer object")
            c = ChainConsumer()
            do_full = False
            biases = {}
            b = 1
            truth = {"$\\Omega_m$": 0.3, "$w\\ \\mathrm{Blinded}$": -1.0, "$\\Omega_\\Lambda$": 0.7}
            shift_params = truth if args.get("SHIFT") else None

            for index, basename in enumerate(args.get("PARSED_FILES")):
                if args.get("NAMES"):
                    name = args.get("NAMES")[index].replace("_", " ")
                else:
                    name = os.path.basename(basename).replace("_", " ")
                # Do smarter biascor
                if ")" in name:
                    key = name.split(")", 1)[1]
                else:
                    key = name
                if key not in biases:
                    biases[key] = b
                    b += 1
                bias_index = biases[key]

                linestyle = "-" if name.lower().endswith("all") else "--"

                weights, likelihood, chain, labels = load_output(basename)
                if args.get("PRIOR"):
                    prior = args.get("PRIOR", 0.01)
                    logging.info(f"Applying prior width {prior} around 0.3")
                    om_index = labels.index("$\\Omega_m$")
                    from scipy.stats import norm

                    prior = norm.pdf(chain[:, om_index], loc=0.3, scale=prior)
                    weights *= prior

                c.add_chain(chain, weights=weights, parameters=labels, name=name, posterior=likelihood, shift_params=shift_params, linestyle=linestyle)

            # Write all our glorious output
            c.configure(plot_hists=False)
            param_bounds = c.analysis.get_summary()  # gets a summary, there's one for each chain, which in this case is each set of COVOPTS


            # I want the best fit for everything combined and then the best fit for each individual one
            # Order: All, X, S, E, C
            fields = ['ALL', 'X', 'S', 'E', 'C']
            fit_params = {}

            for i, pb in enumerate(param_bounds):
                try:
                    fit_params[fields[i]] = (pb['$\\Omega_m\\ \\mathrm{Blinded}$'][1], pb['$w\\ \\mathrm{Blinded}$'][1])
                except KeyError:
                    fit_params[fields[i]] = (pb['$\\Omega_m$'][1], pb['$w$'][1])

            print(fit_params)
            exit()

            m0diff_file = args2.get("M0DIFF_PARSED")
            fitres_files = args2.get("FITRES_PARSED")
            prob_cols = args2.get("FITRES_PROB_COLS")
            # for f, p in zip(fitres_files, prob_cols):
            # make_hubble_plot_simple(f, m0diff_file, best_cosmology=(omega_m_best, w_best))
            make_hubble_plot_combined(fitres_files, m0diff_file, prob_cols, fit_params)


        logging.info("Finishing gracefully")
    except Exception as e:
        logging.exception(str(e))
        raise e
