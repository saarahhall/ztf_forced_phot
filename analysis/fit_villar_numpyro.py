import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist, infer

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pandas as pd

from scipy.stats import median_abs_deviation


def load_lc_df(sn, lc_path, min_num_obs=10):
    """
    Load ZTF light curve (LC) into a dataframe, with quality cuts

    Parameters
    ----------
    sn : string
        ZTF name of the SN to be fit by the model

    lc_path : string (optional, default = '')
        File path to folder containing processed fps lc files from Miller+24

    Returns
    -------
    lc_df_clean : pandas dataframe
        Dataframe of LC obs. with flags==0 and valid flux measurements. Entire
        passbands may be excluded if there were fewer than 10 valid points
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


def calc_sn_exp_both(t, A, B, t0, gamma, trise, tfall, offset):
    f = jnp.where(gamma + t0 >= t,
                  ((A + B * (t - t0)) / (1 + jnp.exp(-(t - t0) / trise))) + offset,
                  offset + ((A + B * gamma) *
                            jnp.exp(-(t - (gamma + t0)) / tfall) /
                            (1 + jnp.exp(-(t - t0) / trise))))
    return f


def lc_model(t_val, Y_unc_val, Y_observed_val=None):
    # sampling...
    # Define priors based on Villar+19
    # x_grid = jnp.logspace(-3, np.log10(60), 500)
    # logistic_pdf = lambda x, tau, x0: 1/(1 + jnp.exp(-(x - x0)/tau))
    trise = numpyro.sample("trise", dist.Uniform(low=0.01, high=50))  # dist.continuous.Uniform?
    # pm.Interpolated('trise', x_grid,
    #                logistic_pdf(x_grid, 5e-3, 0.1))

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

    mixing_dist = dist.Categorical(probs=jnp.array([2 / 3, 1 / 3]))
    component_dist = dist.Normal(loc=jnp.array([5, 60]), scale=jnp.array([5, 30]))
    gamma = numpyro.sample("gamma", dist.MixtureSameFamily(mixing_distribution=mixing_dist,
                                                           component_distribution=component_dist))

    # Expected value of outcome
    mu_switch = calc_sn_exp_both(t_val, Amplitude, Beta, t0, gamma, trise, tfall, scalar)

    # Sample!
    # with numpyro.plate("data", len(t_val)):
    numpyro.sample("y",
                   dist.Normal(mu_switch, Y_unc_val),
                   obs=Y_observed_val)

def lc_model_hier_oneSN(t_val, Y_unc_val, Y_observed_val=None):# find a way to add this -> within numpyro model structure hier_vars=[]):
    # sampling...
    # Define priors based on Villar+19
    # x_grid = jnp.logspace(-3, np.log10(60), 500)
    # logistic_pdf = lambda x, tau, x0: 1/(1 + jnp.exp(-(x - x0)/tau))
    # pm.Interpolated('trise', x_grid,
    #                logistic_pdf(x_grid, 5e-3, 0.1))
    #if 'trise' in hier_vars:
    trise_low = numpyro.sample("trise_low", dist.Uniform(low=5e-3, high=0.02))
    trise_high = numpyro.sample("trise_high", dist.Uniform(low=45, high=55))
    trise = numpyro.sample("trise", dist.Uniform(low=trise_low, high=trise_high)) #0.01,50 # dist.continuous.Uniform?
    #else:
    #    trise = numpyro.sample("trise", dist.Uniform(low=0.01, high=50))

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

    mixing_dist = dist.Categorical(probs=jnp.array([2 / 3, 1 / 3]))
    component_dist = dist.Normal(loc=jnp.array([5, 60]), scale=jnp.array([5, 30]))
    gamma = numpyro.sample("gamma", dist.MixtureSameFamily(mixing_distribution=mixing_dist,
                                                           component_distribution=component_dist))

    # Expected value of outcome
    mu_switch = calc_sn_exp_both(t_val, Amplitude, Beta, t0, gamma, trise, tfall, scalar)

    # Sample!
    # with numpyro.plate("data", len(t_val)):
    numpyro.sample("y",
                   dist.Normal(mu_switch, Y_unc_val),
                   obs=Y_observed_val)


def plot_posterior_draws_numpyro(filt_sampler_pairs, sn, lc_path=''):
    """
    Plot posterior draws of model from Villar+19 to ZTF light curve

    Parameters
    ----------
    sampler: infer.MCMC object
        sampler run using `lc_model`

    sn : string
        ZTF name of the SN to be fit by the model

    lc_path : string (optional, default = '')
        File path to folder containing processed fps lc files from Miller+24
    """
    color_dict = {'ZTF_g': "MediumAquaMarine",
                  'ZTF_r': "Crimson",
                  'ZTF_i': "GoldenRod"}

    lc_df = load_lc_df(sn, lc_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    for pair in filt_sampler_pairs:
        filt, sampler = pair
        pb = 'ZTF_'+filt

        lc_df_thisfilt = lc_df[(lc_df["passband"] == f'ZTF_{filt}')]

        jd0 = 2458119.5  # 2018 Jan 01

        try:
            chains = az.from_numpyro(sampler)  # az.from_netcdf(f"{out_path}/{sn}_{filt}.nc")
        except:
            print(f'Unable to find chains for {sn} in {pb}, skipping this filter.')
        ax.errorbar(lc_df_thisfilt.jd.values - jd0,
                    lc_df_thisfilt.fnu_microJy.values,
                    lc_df_thisfilt.fnu_microJy_unc.values,
                    fmt='o', color=color_dict[pb])

        # posterior samples
        n_samples = len(chains.posterior.t0.values.flatten())
        rand_idx = np.random.choice(range(n_samples),
                                    90, replace=False)
        pi_t0 = chains.posterior.t0.values.flatten()[rand_idx]
        pi_amp = chains.posterior.Amplitude.values.flatten()[rand_idx]
        pi_beta = chains.posterior.Beta.values.flatten()[rand_idx]
        pi_gamma = chains.posterior.gamma.values.flatten()[rand_idx]
        pi_trise = chains.posterior.trise.values.flatten()[rand_idx]
        pi_tfall = chains.posterior.tfall.values.flatten()[rand_idx]
        pi_scalar = chains.posterior.scalar.values.flatten()[rand_idx]

        t_grid_rise = jnp.linspace(pi_t0 - 150,
                                   pi_t0 + pi_gamma,
                                   num=10000)
        ax.plot(t_grid_rise,
                calc_sn_exp_both(t_grid_rise,
                                 pi_amp,
                                 pi_beta,
                                 pi_t0,
                                 pi_gamma,
                                 pi_trise,
                                 pi_tfall,
                                 pi_scalar),
                color=color_dict[pb], ls='--', lw=0.6, alpha=0.3)
        t_grid_decline = jnp.linspace(pi_t0 + pi_gamma,
                                      jnp.max(lc_df_thisfilt.jd.values) - jd0,
                                      num=10000)
        ax.plot(t_grid_decline,
                calc_sn_exp_both(t_grid_decline,
                                 pi_amp,
                                 pi_beta,
                                 pi_t0,
                                 pi_gamma,
                                 pi_trise,
                                 pi_tfall,
                                 pi_scalar),
                color=color_dict[pb], ls='--', lw=0.6, alpha=0.3)

        ax.set_xlabel('Time (JD - 2018 Jan 01)', fontsize=14)
        ax.set_ylim(-3 * median_abs_deviation(lc_df_thisfilt.fnu_microJy.values),
                    1.2 * jnp.percentile(lc_df_thisfilt.fnu_microJy.values, 99.5))
        ax.set_ylabel(r'Flux ($\mu$Jy)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        fig.subplots_adjust(left=0.8, bottom=0.13, right=0.99, top=0.99)
