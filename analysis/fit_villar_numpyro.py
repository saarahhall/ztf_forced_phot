import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist, infer

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pandas as pd
from scipy.stats import median_abs_deviation
from numpyro.infer.initialization import init_to_mean, init_to_uniform

import argparse


def calc_sn_exp_both(t, A, B, t0, gamma, trise, tfall, offset):
    """
    Calculate the rising and declining (hence _both) portions of the SN model from Villar+19

    Parameters
    ----------
    t : array-like
        Times at which the model is evaluated

    A - float
        Constant of proportionality

    B - float
        constant of proportionality

    t0 - float
        reference time for exponential rise

    gamma - float
        time of plateau onset minus t0

    trise - float
        exponential decay factor

    tfall - float
        exponential decay factor controlling the decline

    offset - float
        scalar offset relative to 0 flux

    Returns
    -------
    f : array-like
        Eqn (1) from Villar+19 for flux of SN evaluated at
        times t

    References
    ----------
    (1) V. A. Villar, E. Berger, G. Miller et al. (2019)
    Astrophysical Journal, 884, 83; doi: 10.3847/1538-4357/ab418c
    """
    f = jnp.where(gamma + t0 >= t,
                  ((A + B * (t - t0)) / (1 + jnp.exp(-(t - t0) / trise))) + offset,
                  offset + ((A + B * gamma) *
                            jnp.exp(-(t - (gamma + t0)) / tfall) /
                            (1 + jnp.exp(-(t - t0) / trise))))
    return f


def load_lc_df(sn, lc_path, min_num_obs=10):
    """
    Load ZTF light curve (LC) into a dataframe, with quality cuts

    Parameters
    ----------
    sn : string
        ZTF name of the SN to be fit by the model

    lc_path : string (optional, default = '')
        File path to folder containing processed fps lc files from Miller+24

    min_num_obs : int (optional, default = 10)
        If there are fewer than min_num_obs observations for any given filter,
        exclude from output dataframe

    Returns
    -------
    lc_df_clean : pandas dataframe
        Dataframe of LC obs. with flags==0 and valid flux measurements. Entire
        passbands may be excluded if there were fewer than min_num_obs valid points
    """

    lc_df = pd.read_csv(f"{lc_path}/{sn}_fnu.csv")

    keep_ind = []

    for pb in lc_df.passband.unique():

        pb_lc = np.where((lc_df["passband"] == pb) &
                         (lc_df["flags"] == 0) &
                         # # CHECK IF THIS IS WHAT SHOULD BE DONE!!!
                         # (lc_df["programid"] == 1) &
                         (lc_df["fnu_microJy"] != -999)
                         )
        if len(pb_lc[0]) < min_num_obs:
            # if fewer than min_num_obs pts in this filter, don't include in df
            continue
        else:
            keep_ind.append(pb_lc[0])

    lc_df_clean = lc_df.iloc[np.concatenate(keep_ind)]
    return lc_df_clean


def lc_model(t_val, Y_unc_val, Y_observed_val=None):
    """
    (Non-hierarchical) model for ZTF light curves, with priors loosely based on Villar+19

    Parameters
    ----------
    t_val : array-like
        Time values at which the model is evaluated

    Y_unc_val : array-like
        Flux uncertainties at the observed times t_val

    Y_observed_val : array-like (optional, default = None)
        Flux observations at times t_val
    """

    # Define priors based on Villar+19
    trise = numpyro.sample("trise", dist.Uniform(low=0.01, high=50))  # dist.continuous.Uniform?
    tfall = numpyro.sample("tfall", dist.Uniform(low=1, high=300))

    Amp_Guess = jnp.max(Y_observed_val)
    Amplitude = numpyro.sample("Amplitude", dist.TruncatedNormal(
        loc=Amp_Guess,
        scale=Amp_Guess / 10, low=0))

    Beta = numpyro.sample("Beta", dist.Uniform(low=-jnp.max(Y_observed_val) / 150, high=0))

    t0 = numpyro.sample("t0", dist.Uniform(
        low=jnp.array(t_val)[jnp.argmax(Y_observed_val)] - 25,
        high=jnp.array(t_val)[jnp.argmax(Y_observed_val)] + 25))

    sigma_est = jnp.sqrt(jnp.mean(Y_unc_val ** 2))
    scalar = numpyro.sample("scalar", dist.TruncatedNormal(loc=0, scale=sigma_est,
                                                           low=-2 * sigma_est,
                                                           high=2 * sigma_est))

    ## gamma's prior is a normal mixture - not straightforward in numpyro
    # Define the weights for the mixture components
    weights = jnp.array([2 / 3, 1 / 3])  # Should sum to 1
    # Define means and standard deviations for the normal components
    means, stds = jnp.array([5, 60]), jnp.array([5, 30])
    # Define the normal distributions
    components = [dist.Normal(mu, sigma) for mu, sigma in zip(means, stds)]
    # Define the mixture distribution
    mixture = dist.Mixture(
        dist.Categorical(probs=weights),  # Mixture weights
        components  # List of component distributions
    )
    gamma = numpyro.sample("gamma", mixture)


    # Expected value of outcome
    mu_switch = calc_sn_exp_both(t_val, Amplitude, Beta, t0, gamma, trise, tfall, scalar)

    # Sample!
    numpyro.sample("y",
                   dist.Normal(mu_switch, Y_unc_val),
                   obs=Y_observed_val)

def fit_gr_numpyro(sn, lc_path, out_path, num_warmup=15000, num_samples=1000, num_chains=4, init_strat='uniform', model=lc_model):
    """
    Fit parametric model from Villar+19 to ZTF light curve

    Parameters
    ----------
    sn : string
        ZTF name of the SN to be fit by the model

    lc_path : string (optional, default = '')
        File path to folder containing processed fps lc files from Miller+24

    out_path : string (optional, default = '')
        File path to write output files (MCMC chains and summary statistics)

    num_warmup : int (optional, default = 1500)
        Number of warmup steps to run MCMC chains for

    num_samples : int (optional, default = 1000)
        Number of samples to run MCMC chains for

    num_chains : int (optional, default = 2)
        Number of chains to run MCMC, make sure num_chains =< jax.local_device_count()
        ( you can change device count using: numpyro.set_host_device_count(#) )

    init_strat : str (optional, default = 'uniform')
        Number of chains to run MCMC, make sure num_chains =< jax.local_device_count()
        ( you can change device count using: numpyro.set_host_device_count(#) )

    model : function (optional, default = lc_model)
        numpyro modeling function, either hierarchical (WIP) or non-hierarchical (lc_model)
    """

    # load dataframe with some quality cuts
    lc_df = load_lc_df(sn=sn, lc_path=lc_path)

    for pb in lc_df.passband.unique():
        filt = pb[-1]

        # Fit the model with the current filter
        lc_df_thisfilt = lc_df[(lc_df["passband"] == f'ZTF_{filt}')]

        jd0 = 2458119.5  # 2018 Jan 01
        time_axis = (lc_df_thisfilt['jd'].values) - jd0

        Y_observed = ((lc_df_thisfilt['fnu_microJy']).values)

        Y_unc = ((lc_df_thisfilt['fnu_microJy_unc']).values)

        if init_strat == 'uniform':
            init_strategy = init_to_uniform
        elif init_strat == 'mean':
            init_strategy = init_to_mean
        else:
            print('could not define initialization strategy based on input!')

        sampler = infer.MCMC(
            infer.NUTS(model, init_strategy = init_strategy),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True)
        sampler.warmup(jax.random.PRNGKey(0), time_axis, Y_unc, Y_observed_val=Y_observed, collect_warmup=True)
        warmup_samples = sampler.get_samples()
        np.save(f'{out_path}/{sn}_{filt}_warmupsamples.npy', warmup_samples)

        # Draw samples from the posterior
        sampler.run(jax.random.PRNGKey(0), time_axis, Y_unc, Y_observed_val=Y_observed)
        samples = sampler.get_samples()
        np.save(f'{out_path}/{sn}_{filt}_samples.npy', samples)

        # Save results
        data = az.from_numpyro(sampler)
        data.to_netcdf(f"{out_path}/{sn}_{filt}_numpyro.nc")
        out_summary = f"{out_path}/{sn}_{filt}_numpyro.csv"
        az.summary(data, stat_focus='median').to_csv(out_summary)


def lc_model_hier_oneSN(t_val, Y_unc_val, Y_observed_val=None):
    """
    Work in progress!!
    Hierarchical model for ZTF light curves, with priors loosely based on Villar+19

    Parameters
    ----------
    t_val : array-like
        Time values at which the model is evaluated

    Y_unc_val : array-like
        Flux uncertainties at the observed times t_val

    Y_observed_val : array-like (optional, default = None)
        Flux observations at times t_val
    """

    # Define priors based on Villar+19
    trise_low = numpyro.sample("trise_low", dist.Uniform(low=5e-3, high=0.02))
    trise_high = numpyro.sample("trise_high", dist.Uniform(low=45, high=55))
    trise = numpyro.sample("trise", dist.Uniform(low=trise_low, high=trise_high)) #0.01,50 # dist.continuous.Uniform?


    tfall_low = numpyro.sample("tfall_low", dist.Uniform(low=0.5, high=1.5))
    tfall_high = numpyro.sample("tfall_high", dist.Uniform(low=295, high=305))
    tfall = numpyro.sample("tfall", dist.Uniform(low=tfall_low, high=tfall_high))


    Amp_Guess = jnp.max(Y_observed_val)
    Amp_mu = numpyro.sample("Amp_mu", dist.Uniform(low=Amp_Guess-500, high=Amp_Guess+500))
    #Amp_sigma = numpyro.sample("Amp_sigma", dist.Uniform(low=Amp_Guess/11, high=Amp_Guess/9))
    Amplitude = numpyro.sample("Amplitude", dist.TruncatedNormal(
        loc=Amp_mu,
        scale=Amp_Guess / 10, low=0))
        #loc=Amp_Guess,
        #scale=Amp_Guess / 10, low=0))

    Beta = numpyro.sample("Beta", dist.Uniform(low=-jnp.max(Y_observed_val) / 150, high=0))

    t0 = numpyro.sample("t0", dist.Uniform(
        low=jnp.array(t_val)[jnp.argmax(Y_observed_val)] - 25,
        high=jnp.array(t_val)[jnp.argmax(Y_observed_val)] + 25))

    sigma_est = jnp.sqrt(jnp.mean(Y_unc_val ** 2))
    scalar = numpyro.sample("scalar", dist.TruncatedNormal(loc=0, scale=sigma_est,
                                                           low=-2 * sigma_est,
                                                           high=2 * sigma_est))

    ## gamma's prior is a normal mixture - not straightforward in numpyro
    # Define the weights for the mixture components
    weights = jnp.array([2 / 3, 1 / 3])  # Should sum to 1
    # Define means and standard deviations for the normal components
    means, stds = jnp.array([5, 60]), jnp.array([5, 30])
    # Define the normal distributions
    components = [dist.Normal(mu, sigma) for mu, sigma in zip(means, stds)]
    # Define the mixture distribution
    mixture = dist.Mixture(
        dist.Categorical(probs=weights),  # Mixture weights
        components  # List of component distributions
    )
    gamma = numpyro.sample("gamma", mixture)

    # Expected value of outcome
    mu_switch = calc_sn_exp_both(t_val, Amplitude, Beta, t0, gamma, trise, tfall, scalar)

    # Sample!
    # with numpyro.plate("data", len(t_val)):
    numpyro.sample("y",
                   dist.Normal(mu_switch, Y_unc_val),
                   obs=Y_observed_val)


def plot_posterior_draws_numpyro(sn, lc_path='', out_path='', save_fig=True):
    """
    Plot posterior draws of model from Villar+19 to ZTF light curve

    Parameters
    ----------
    sn : string
        ZTF name of the SN to be fit by the model

    lc_path : string (optional, default = '')
        File path to folder containing processed fps lc files from Miller+24

    out_path : string (optional, default = '')
        File path to files saved by fit_gr()  (MCMC chains and summary statistics)

    save_fig : boolean (optional, default = True)
        Boolean flag indicating whether or not to save the plot as a png
    """
    color_dict = {'ZTF_g': "MediumAquaMarine",
                  'ZTF_r': "Crimson",
                  'ZTF_i': "GoldenRod"}

    lc_df = load_lc_df(sn, lc_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    for pb in lc_df.passband.unique():
        filt = pb[-1]

        lc_df_thisfilt = lc_df[(lc_df["passband"] == f'ZTF_{filt}')]

        jd0 = 2458119.5  # 2018 Jan 01

        try:
            chains = az.from_netcdf(f"{out_path}/{sn}_{filt}_numpyro.nc")
        except:
            print(f'Unable to find chains for {sn} in {pb}, skipping this filter.')
            continue
        ax.errorbar(lc_df_thisfilt.jd.values - jd0,
                    lc_df_thisfilt.fnu_microJy.values,
                    lc_df_thisfilt.fnu_microJy_unc.values,
                    fmt='o', color=color_dict[pb])

        # posterior samples
        n_samples = len(chains.posterior.t0.values.flatten())
        rand_idx = np.random.choice(range(n_samples),
                                    10, replace=False)
        pi_t0 = chains.posterior.t0.values.flatten()[rand_idx]
        pi_amp = chains.posterior.Amplitude.values.flatten()[rand_idx]
        pi_beta = chains.posterior.Beta.values.flatten()[rand_idx]
        pi_gamma = chains.posterior.gamma.values.flatten()[rand_idx]
        pi_trise = chains.posterior.trise.values.flatten()[rand_idx]
        pi_tfall = chains.posterior.tfall.values.flatten()[rand_idx]
        pi_scalar = chains.posterior.scalar.values.flatten()[rand_idx]

        t_grid = jnp.linspace(pi_t0 - 150,
                              jnp.max(lc_df_thisfilt.jd.values) - jd0,
                              num=20000)
        ax.plot(t_grid,
                calc_sn_exp_both(t_grid,
                                 pi_amp,
                                 pi_beta,
                                 pi_t0,
                                 pi_gamma,
                                 pi_trise,
                                 pi_tfall,
                                 pi_scalar),
                color=color_dict[pb], ls='--', lw=0.6, alpha=0.3)

        ax.set_xlabel('Time (JD - 2018 Jan 01)', fontsize=14)
        # xlim may be unstable, since we're picking a random posterior instead of the max. prob. posterior
        x_max = np.min([pi_t0[0] + pi_gamma[0] + 10 * pi_tfall[0],
                        np.max(lc_df_thisfilt.jd.values) - jd0 + 10])
        ax.set_xlim(pi_t0[0] - 75, x_max)
        ax.set_ylim(-3 * median_abs_deviation(lc_df_thisfilt.fnu_microJy.values),
                    1.2 * jnp.percentile(lc_df_thisfilt.fnu_microJy.values, 99.5))
        ax.set_ylabel(r'Flux ($\mu$Jy)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        #fig.subplots_adjust(left=0.8, bottom=0.13, right=0.99, top=0.99)
        if save_fig:
            fig.savefig(f"{lc_path}/{sn}_posterior_numpyro.png",
                        dpi=600, transparent=True)

def main():

    # Initialize argument parser
    parser = argparse.ArgumentParser(prog='fit_villar_numpyro.py <sn>',
                                     description='Run Villar+19 light curve fits on a BTS fps lc. with numpyro')
    # Necessary arguments
    parser.add_argument('sn', type=str, nargs='?', default=None,
                   help='ZTF transient name')
    # Optional arguments
    parser.add_argument('lc_path', type=str, nargs='?', default=None,
                   help='path to folder containing processed fps lc file from Miller+24')
    parser.add_argument('out_path', type=str, nargs='?', default=None,
                   help='path for output MCMC chains')
    parser.add_argument('num_warmup', type=int, nargs='?', default=None,
                        help='number of warmup steps for MCMC')
    parser.add_argument('num_samples', type=int, nargs='?', default=None,
                        help='number of samples for MCMC')
    parser.add_argument('num_chains', type=int, nargs='?', default=None,
                        help='number of chains for MCMC')
    parser.add_argument('init_strat', type=str, nargs='?', default=None,
                        help='initialization strategy for MCMC')

    try:
        args = parser.parse_args()

        run = True
    except Exception:
        run = False

    if run:
        fit_gr_numpyro(**vars(args))

if __name__ == "__main__":
    main()