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
            print(corrected_shift)
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
        print('\n')
        print(field_letter, om, w)

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
        print(corrected_shift)

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


if __name__ == "__main__":
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
