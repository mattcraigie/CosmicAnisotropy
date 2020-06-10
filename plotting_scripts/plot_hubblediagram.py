import numpy as np
import yaml
from chainconsumer import ChainConsumer
import pandas as pd
import sys
import argparse
import os
import logging


def fail(msg, condition=True):
    if condition:
        logging.error(msg)
        raise ValueError(msg)


def setup_logging():
    fmt = "[%(levelname)8s |%(filename)21s:%(lineno)3d]   %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=[handler, logging.FileHandler(
        "plot_cosmomc.log")])
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

    name, sim_num, *_ = fitres_file.split("_")
    sim_num = int(sim_num)

    # load fitres and m0diff dataframes
    df = pd.read_csv(fitres_file, delim_whitespace=True, comment="#")
    dfm = pd.read_csv(m0diff_file)

    # select only rows with the right name, sim num, and a zero muopt num and fitopt_num (what are these?)
    dfm = dfm[(dfm.name == name) & (dfm.sim_num == sim_num) & (dfm.muopt_num == 0) & (dfm.fitopt_num == 0)]

    from astropy.cosmology import FlatwCDM
    import numpy as np
    import matplotlib.pyplot as plt

    df.sort_values(by="zHD", inplace=True)
    dfm.sort_values(by="z", inplace=True)

    dfm = dfm[dfm["MUDIFERR"] < 10]  # select only rows with MUDIFERR < 10 (what is that?)

    # get the first unique value of omega lambda and w
    ol = dfm.ol_ref.unique()[0]
    w = dfm.w_ref.unique()[0]
    alpha = 0
    beta = 0
    sigint = 0
    gamma = r"$\gamma = 0$"
    scalepcc = "NA"
    num_sn_fit = df.shape[0]
    contam_data, contam_true = "", ""

    # getting the values of a bunch of params and formatting them nicely
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

    # get distmods across the range of the SN according to flat w CDM
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

    # if the lowest z value is less than the default n threshold then  lower n_space and make subsec False
    if zs.min() > n_thresh:
        n_space = 0.01
        subsec = False

    # set up a log space from 0.01 to n_thresh
    z_a = np.logspace(np.log10(min(0.01, zs.min() * 0.9)), np.log10(n_thresh), int(n_space * n_trans))

    # set up a linear space from n_thresh upwards
    z_b = np.linspace(n_thresh, zs.max() * 1.01, 1 + int((1 - n_space) * n_trans))[1:]

    # join the spaces
    z_trans = np.concatenate((z_a, z_b))
    z_scale = np.arange(n_trans)

    # create an interpolation function for any zs
    def tranz(zs):
        return interp1d(z_trans, z_scale)(zs)

    # set up plotting ticks
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

    # put in data
    for resid, ax in enumerate(axes):
        ax.tick_params(which="major", direction="inout", length=4)
        ax.tick_params(which="minor", direction="inout", length=3)
        if resid:
            # This is set if it's the residual plot
            sub = df["MUMODEL"]  # get mu from file
            sub2 = 0
            sub3 = distmod  # get the flat w cdm model expected mu
            ax.set_ylabel(r"$\Delta \mu$")
            ax.tick_params(top=True, which="both")
            alpha = 0.2
            ax.set_ylim(-0.5, 0.5)
        else:
            # This is set if it's the normal Hubble diagram
            sub = 0
            sub2 = -dfm["MUREF"]  # mu ref? what is this? Apparently just the normal mu?
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

        # Plot each supernova
        ax.errorbar(tranz(dfz), df["MU"] - sub, yerr=df["MUERR"], fmt="none", elinewidth=0.5, c="#AAAAAA", alpha=0.5 * alpha)
        h = ax.scatter(tranz(dfz), df["MU"] - sub, c=cc, s=1, zorder=2, alpha=alpha, vmax=vmax, cmap=cmap)

        #if not args.blind:  # Gives us a fit even if we have blinded it
        if True:
            logging.info("Plotting in cosmology")
            # Plot ref cosmology
            ax.plot(tranz(zs), distmod - sub3, c="k", zorder=-1, lw=0.5, alpha=0.7)

            # Plot m0diff
            ax.errorbar(tranz(dfm["z"]), dfm["MUDIF"] - sub2, yerr=dfm["MUDIFERR"], fmt="o", mew=0.5, capsize=3, elinewidth=0.5, c="k", ms=4)

        # Here will be where I plot the best fit cosmology

        ax.plot()
        ax.set_xticks(x_tick_t)
        ax.set_xticks(x_ticks_mt, minor=True)
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(z_scale.min(), z_scale.max())

        if args.blind:
            ax.set_yticklabels([])
            ax.set_yticks([])

    # set up a colour bar based on the probability that it's a Ia SN
    if color_prob:
        cbar = fig.colorbar(h, ax=axes, orientation="vertical", fraction=0.1, pad=0.01, aspect=40)
        cbar.set_label("Prob Ia")

    # saving figure
    fp = fitres_file.replace(".fitres", ".png")
    fp = fp.replace("FITOPT0_MUOPT0", "HUBBLE_PLOT")
    logging.debug(f"Saving Hubble plot to {fp}")
    fig.savefig(fp, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    setup_logging()
    args = get_arguments()  # loads in the yml file

    try:
        if args.get("PARSED_FILES"):
            logging.info("Creating chain consumer object")
            c = ChainConsumer()
            do_full = False
            biases = {}
            b = 1
            truth = {"$\\Omega_m$": 0.3, "$w\\ \\mathrm{Blinded}$": -1.0, "$\\Omega_\\Lambda$": 0.7}
            shift_params = truth if args.get("SHIFT") else None

            for index, basename in enumerate(args.get("PARSED_FILES")):  # getting each name
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

                # grab chain data
                weights, likelihood, chain, labels = load_output(basename)

                # do prior things
                if args.get("PRIOR"):
                    prior = args.get("PRIOR", 0.01)
                    logging.info(f"Applying prior width {prior} around 0.3")
                    om_index = labels.index("$\\Omega_m$")
                    from scipy.stats import norm

                    prior = norm.pdf(chain[:, om_index], loc=0.3, scale=prior)
                    weights *= prior

                # add chain data  to ChainConsumer
                c.add_chain(chain, weights=weights, parameters=labels, name=name, posterior=likelihood, shift_params=shift_params, linestyle=linestyle)

            # plotting using ChainConsumer default plotting stuff
            # Write all our glorious output

            c.configure(plot_hists=False)

            param_bounds = c.analysis.get_summary()  # gets a summary, there's one for each chain, which in this case is each set of COVOPTS

            # these are the param outputs from the ALL chain. Weird keys but what can ya do.
            omega_m_best = param_bounds[0]['$\\Omega_m\\ \\mathrm{Blinded}$'][1]  # the value in position one is the max likelihood
            w_best = param_bounds[0]['$w\\ \\mathrm{Blinded}$'][1]

            logging.info("Hello something is happening")

            # perhaps I can do this more generally for any model... but plot_hubble is only set up for two vars.
            # also, I can't fit H0.

                        # Plot hubble diagrams
            m0diff_file = args.get("M0DIFF_PARSED")  # data
            fitres_files = args.get("FIIRES_PARSED")  # no data
            prob_cols = args.get("FITRES_PROB_COLS")  # no data

            for f, p in zip(fitres_files, prob_cols):
                make_hubble_plot(f, m0diff_file, p, args, best_cosmology=(omega_m_best, w_best))

        logging.info("Finishing gracefully")
    except Exception as e:
        logging.exception(str(e))
        raise e