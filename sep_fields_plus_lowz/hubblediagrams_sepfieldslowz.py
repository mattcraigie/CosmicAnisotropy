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


def make_hubble_plot(fitres_file, m0diff_file, prob_col_name, args, best_cosmology):
    logging.info("Making Hubble plot")
    # Note that the fitres file has mu and fit 0, m0diff will have to select down to it
    print(fitres_file)
    field_letter, field, sim_num, *_ = fitres_file.split("_")
    name = field_letter + field
    sim_num = int(sim_num)

    df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")
    dfm = pd.read_csv(m0diff_file)
    dfm = dfm[(dfm.name == name) & (dfm.sim_num == sim_num) & (dfm.muopt_num == 0) & (dfm.fitopt_num == 0)]

    from astropy.cosmology import FlatwCDM
    import numpy as np
    import matplotlib.pyplot as plt

    df.sort_values(by="zHD", inplace=True)
    dfm.sort_values(by="z", inplace=True)
    dfm = dfm[dfm["MUDIFERR"] < 10]

    alpha = 0
    beta = 0
    sigint = 0
    gamma = r"$\gamma = 0$"
    scalepcc = "NA"
    num_sn_fit = df.shape[0]
    contam_data, contam_true = "", ""

    with open(fitres_file) as f:
        for line in f.read().splitlines():
            if "NSNFIT" in line:
                v = int(line.split("=", 1)[1].strip())
                num_sn_fit = v
                num_sn = f"$N_{{SN}} = {v}$"
            if "alpha0" in line and "=" in line and "+-" in line:
                alpha = r"$\alpha = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
            if "beta0" in line and "=" in line and "+-" in line:
                beta = r"$\beta = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
            if "sigint" in line and "iteration" in line:
                sigint = r"$\sigma_{\rm int} = " + line.split()[3] + "$"
            if "gamma" in line and "=" in line and "+-" in line:
                gamma = r"$\gamma = " + line.split("=")[-1].replace("+-", r"\pm") + "$"
            if "CONTAM_TRUE" in line:
                v = max(0.0, float(line.split("=", 1)[1].split("#")[0].strip()))
                n = v * num_sn_fit
                contam_true = f"$R_{{CC, true}} = {v:0.4f} (\\approx {int(n)} SN)$"
            if "CONTAM_DATA" in line:
                v = max(0.0, float(line.split("=", 1)[1].split("#")[0].strip()))
                n = v * num_sn_fit
                contam_data = f"$R_{{CC, data}} = {v:0.4f} (\\approx {int(n)} SN)$"
            if "scalePCC" in line and "+-" in line:
                scalepcc = "scalePCC = $" + line.split("=")[-1].strip().replace("+-", r"\pm") + "$"
    prob_label = prob_col_name.replace("PROB_", "").replace("_", " ")
    label = "\n".join([num_sn, alpha, beta, sigint, gamma, scalepcc, contam_true, contam_data, f"Classifier = {prob_label}"])
    label = label.replace("\n\n", "\n").replace("\n\n", "\n")
    dfz = df["zHD"]
    zs = np.linspace(dfz.min(), dfz.max(), 500)

    ### ~~~ SET COSMOLOGY HERE ~~~ ###
    # so it looks like omega_lambda and w have been included in the data itself. Not sure how to tackle this.
    # distmod = FlatwCDM(70, 1 - ol, w).distmod(zs).value
    om, w = best_cosmology
    logging.info("Best fit cosmology is om={} and w={}".format(om, w))
    distmod = FlatwCDM(H0=70, Om0=om, w0=w).distmod(zs).value


    n_trans = 1000
    n_thresh = 0.05
    n_space = 0.3
    subsec = True
    if zs.min() > n_thresh:
        n_space = 0.01
        subsec = False
    z_a = np.logspace(np.log10(min(0.01, zs.min() * 0.9)), np.log10(n_thresh), int(n_space * n_trans))
    z_b = np.linspace(n_thresh, zs.max() * 1.01, 1 + int((1 - n_space) * n_trans))[1:]
    z_trans = np.concatenate((z_a, z_b))
    z_scale = np.arange(n_trans)

    def tranz(zs):
        return interp1d(z_trans, z_scale)(zs)

    if subsec:
        x_ticks = np.array([0.01, 0.02, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
        x_ticks_m = np.array([0.03, 0.04, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9])
    else:
        x_ticks = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
        x_ticks_m = np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.9])
    mask = (x_ticks > z_trans.min()) & (x_ticks < z_trans.max())
    mask_m = (x_ticks_m > z_trans.min()) & (x_ticks_m < z_trans.max())
    x_ticks = x_ticks[mask]
    x_ticks_m = x_ticks_m[mask_m]
    x_tick_t = tranz(x_ticks)
    x_ticks_mt = tranz(x_ticks_m)

    fig, axes = plt.subplots(figsize=(7, 5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0})

    for resid, ax in enumerate(axes):
        ax.tick_params(which="major", direction="inout", length=4)
        ax.tick_params(which="minor", direction="inout", length=3)
        if resid:
            sub = df["MUMODEL"]
            sub2 = 0
            sub3 = distmod
            ax.set_ylabel(r"$\Delta \mu$")
            ax.tick_params(top=True, which="both")
            alpha = 0.2
            ax.set_ylim(-0.5, 0.5)
        else:
            sub = 0
            sub2 = -dfm["MUREF"]
            sub3 = 0
            ax.set_ylabel(r"$\mu$")
            ax.annotate(label, (0.98, 0.02), xycoords="axes fraction", horizontalalignment="right", verticalalignment="bottom", fontsize=8)
            alpha = 0.7

        ax.set_xlabel("$z$")
        if subsec:
            ax.axvline(tranz(n_thresh), c="#888888", alpha=0.4, zorder=0, lw=0.7, ls="--")

        if df[prob_col_name].min() >= 1.0:
            cc = df["IDSURVEY"]
            vmax = None
            color_prob = False
            cmap = "rainbow"
        else:
            cc = df[prob_col_name]
            vmax = 1.05
            color_prob = True
            cmap = "inferno"

        # Plot each point
        ax.errorbar(tranz(dfz), df["MU"] - sub, yerr=df["MUERR"], fmt="none", elinewidth=0.5, c="#AAAAAA", alpha=0.5 * alpha)
        h = ax.scatter(tranz(dfz), df["MU"] - sub, c=cc, s=1, zorder=2, alpha=alpha, vmax=vmax, cmap=cmap)

        # if not args.get("BLIND", []):  # This line stops cosmo plotting if it's blinded
        if True:
            # Plot ref cosmology
            ax.plot(tranz(zs), distmod - sub3, c="k", zorder=-1, lw=0.5, alpha=0.7)

            # Plot m0diff
            ax.errorbar(tranz(dfm["z"]), dfm["MUDIF"] - sub2, yerr=dfm["MUDIFERR"], fmt="o", mew=0.5, capsize=3, elinewidth=0.5, c="k", ms=4)
        ax.set_xticks(x_tick_t)
        ax.set_xticks(x_ticks_mt, minor=True)
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(z_scale.min(), z_scale.max())


        if args.get("BLIND", []):
            ax.set_yticklabels([])
            ax.set_yticks([])

    color_prob = False
    if color_prob:
        cbar = fig.colorbar(h, ax=axes, orientation="vertical", fraction=0.1, pad=0.01, aspect=40)
        cbar.set_label("Prob Ia")

    fp = fitres_file.replace(".fitres", "_mod.png")
    logging.debug(f"Saving Hubble plot to {fp}")
    fig.savefig(fp, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


def make_hubble_plot_simple(fitres_file, m0diff_file, prob_col_name, args, best_cosmology):
    cdict = {'X': 'red', 'C': 'violet', 'S': 'gold', 'E': 'navy'}

    logging.info("Making Hubble plot")
    # Loading data stuff I don't understand
    field_letter, field, sim_num, *_ = fitres_file.split("_")
    name = field_letter + field
    sim_num = int(sim_num)
    df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")
    dfm = pd.read_csv(m0diff_file)
    dfm = dfm[(dfm.name == name) & (dfm.sim_num == sim_num) & (dfm.muopt_num == 0) & (dfm.fitopt_num == 0)]

    from astropy.cosmology import FlatwCDM
    import numpy as np
    import matplotlib.pyplot as plt
    df.sort_values(by="zHD", inplace=True)
    dfm.sort_values(by="z", inplace=True)
    dfm = dfm[dfm["MUDIFERR"] < 10]

    # best fit lines
    dfz = df["zHD"]
    zs = np.linspace(0.01, 1.00, 500)
    om, w = best_cosmology
    logging.info("Best fit cosmology is om={} and w={}".format(om, w))
    distmod = FlatwCDM(H0=100, Om0=om, w0=w).distmod(zs).value

    # idk what this does but I need the tranz function for later. What on earth is this?

    # skip tick marks

    fig, axes = plt.subplots(figsize=(7, 5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0})

    # strange, strange way to plot

    for resid, ax in enumerate(axes):
        if resid:
            sub = df["MUMODEL"]
            sub2 = 0
            sub3 = distmod
            ax.set_ylabel(r"$\Delta \mu$")
            ax.tick_params(top=True, which="both")
            alpha = 0.2
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel("$z$")
        else:
            sub = 0
            sub2 = -dfm["MUREF"]
            sub3 = 0
            ax.set_ylabel(r"$\mu$")
            alpha = 0.7




        # Plot each point
        ax.errorbar(dfz, df["MU"] - sub, yerr=df["MUERR"], fmt="none", elinewidth=0.5, c="#AAAAAA",
                    alpha=0.5 * alpha)

        h = ax.scatter(dfz, df["MU"] - sub, c=cdict[field_letter], s=1, zorder=2, alpha=1)

        # if not args.get("BLIND", []):  # This line stops cosmo plotting if it's blinded
        if True:
            # Plot ref cosmology
            ax.plot(zs, distmod - sub3, c="k", zorder=-1, lw=0.5, alpha=0.7)

            # Plot m0diff
            ax.errorbar(dfm["z"], dfm["MUDIF"] - sub2, yerr=dfm["MUDIFERR"], fmt="o", mew=0.5, capsize=3,
                        elinewidth=0.5, c="k", ms=4)

        ax.set_xlim(0.05, 1.05)

    fp = fitres_file.replace(".fitres", "_mod.png")
    logging.debug(f"Saving Hubble plot to {fp}")
    fig.savefig(fp, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)

    return


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
            om, w = fit_params['ALL']
            H0_range = np.linspace(60, 70, 101)
            best_H0_all = get_best_H0(H0_range, z, mu, om, w)


            ## 1/n sum(d_i - (best fit)_i)
            # then rescale by mu = mu_original - mu_diff

            zs = np.linspace(0.01, 1.1, 500)
            all_distmod = FlatwCDM(H0=best_H0_all, Om0=om, w0=w).distmod(zs).value



            main_ax.plot(zs, all_distmod, c="k", zorder=-1, lw=0.5, alpha=0.7)
            resid_ax.plot(zs, all_distmod - all_distmod, c="k", zorder=-1, lw=0.5, alpha=0.7)
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

        H0_range = np.linspace(60, 70, 101)
        best_H0 = get_best_H0(H0_range, dfz, df['MU'], om, w)

        zs = np.linspace(0.01, 1.1, 500)
        distmod = FlatwCDM(H0=best_H0, Om0=om, w0=w).distmod(zs).value



        # field cosmology plotting
        main_ax.plot(zs, distmod, c=cdict[field_letter], linestyle=':', zorder=3, lw=0.5, alpha=1)
        resid_ax.plot(zs, distmod - all_distmod, c=cdict[field_letter], linestyle=':', zorder=3, lw=1.5, alpha=1)

    handles = [mpatches.Patch(color=cdict[l], label=l + ' fields') for l in ['X', 'S', 'E', 'C']]
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


def get_best_H0(H0_range, z, mu, om, w):
    scores = []

    for H0 in H0_range:
        predicted_distmod = FlatwCDM(H0=H0, Om0=om, w0=w).distmod(z).value

        mse = np.sum((predicted_distmod - mu) ** 2)

        scores.append(mse)

    return H0_range[np.argmin(scores)]


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
