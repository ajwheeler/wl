"""sample and construct corner plots of priors"""
import priors
import emcee
import model
import numpy as np
import platform
from itertools import compress
import corner
import mcmc

labels = [r"$R_{disk}$", r"$F_{disk}$", r"$g_1^{disk}$", r"$g_2^{disk}$",
          r"$R_{bulge}$", r"$F_{bulge}$", r"$g_1^{bulge}$", r"$g_2^{bulge}$",
          r"$g_1^{shear}$", r"$g_2^{shear}$", r"$\mu$"]

kmask = [False]*4 + [True]*2 + [False]*5
omask = [False]*2 + [True]*2 + [False]*7

def lnprob(theta, name, mask):
    if not mcmc.within_bounds(theta, mask):
        return -np.inf

    params = model.EggParams()
    params.fromArray(theta, mask)
    theta = params.toArray()
    weights = priors.calculate_priors([theta])
    return np.log(weights[name][0])o

nwalkers = 1000
nburnin = 100
nsample = 1000

k = (kmask, "kormendy")
o = (omask, "orientation")

for mask, name in [k,o]:
    ndim = sum(mask)
    theta0 = [model.EggParams().toArray(mask) + 1e-2*np.random.randn(ndim)\
            for _ in range(nwalkers)]
    nthreads = 16 if "cosmos5" in platform.node() else 1

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads,
                                    args=[name, mask])

    pos, _, state = sampler.run_mcmc(theta0, nburnin)
    sampler.reset()
    sampler.run_mcmc(pos, nsample, rstate0=state)

    print(name)
    print(sampler.get_autocorr_time())

    chain = sampler.flatchain
    fig = corner.corner(chain, labels = list(compress(labels, mask)))
    fig.savefig(name + "_corner.png")
