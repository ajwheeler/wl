import model
import platform
import os
import mcmc
import pickle
import numpy as np
import priors
import itertools

def valid(theta):
    """returns True iff all shear vectors have mag < 1"""
    return np.sqrt(theta.g1d**2 + theta.g2d**2) < .9 \
       and np.sqrt(theta.g1b**2 + theta.g2b**2) < .9

def draw_thetas(n):
    """generate n random thetas"""
    #randomly draw initial thetas
    thetas = []
    for i in xrange(n):
        ub = mcmc.theta_ub
        lb = mcmc.theta_lb
        theta = model.EggParams()

        #hacky do-while construct
        badTheta = True
        while badTheta:
            theta.fromArray(np.array([np.random.uniform(low=l, high=u) for l,u in zip(lb,ub)]))
            if valid(theta):
                badTheta = False

        thetas.append(theta)
    print("Thetas generated")
    print(thetas)

nthreads = 16 if "cosmos5" in platform.node() else 1

mask = [True]*8 + [False]*3

nwalkers = 1000
nburnin = 300
nsample = 1000

#parameters to test
SNRs = [10,50,200]
PSFs = [.25,.8]
thetas = [model.EggParams(g1d=.2, g2d=.3, g2b=.4)]
#params is NOT a list of EggParams, its a list of SNR,PSF,EggParams triples
#in this file, the EggParams are called theta
params = list(itertools.product(SNRs,PSFs,thetas))

for i in xrange(len(params)):
    os.mkdir(str(i))
print("dirs created")

for i,(SNR,psf,theta) in enumerate(params):
    print("TODO: use pdf")
    #run single band chain
    data, pixel_var = mcmc.generate_data(theta, dual_band=False, SNR=SNR)

    sampler_s, stats_s = mcmc.run_chain(data, pixel_var, theta, nwalkers,
                                        nburnin, nsample, nthreads=nthreads,
                                        mask=mask, SNR=SNR, psf=psf)

    #save stats, chain, and logprobs
    with open(str(i) + '/single.stats.p', 'wb') as f:
        pickle.dump(stats_s,f)
    np.save(str(i) + '/single.chain.npy', sampler_s.flatchain)
    np.save(str(i) + '/single.lnprob.npy', sampler_s.flatlnprobability)

    #calculate single band priors
    weights = priors.calculate_priors(sampler_s.flatchain)
    for p, ws in weights.iteritems():
        np.save(str(i) + '/single.' + p + '.npy', ws)

    #run dual-band chain
    data, pixel_var = mcmc.generate_data(theta, dual_band=True, SNR=SNR)
    sampler_d, stats_d = mcmc.run_chain(data, pixel_var, theta, nwalkers,
                                        nburnin, nsample, nthreads=nthreads,
                                        mask=mask, SNR=SNR, psf=psf,
                                        dual_band=True)

    with open(str(i) + '/double.stats.p', 'wb') as f:
        pickle.dump(stats_d,f)
    np.save(str(i) + '/double.chain.npy', sampler_d.flatchain)
    np.save(str(i) + '/double.lnprob.npy', sampler_d.flatlnprobability)

    #calculate dual-band priors
    weights = priors.calculate_priors(sampler_d.flatchain)
    for p, ws in weights.iteritems():
        np.save(str(i) + '/double.' + p + '.npy', ws)
