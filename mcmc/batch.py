import model
import platform
import os
import mcmc
import pickle
import numpy as np
import priors

SNR = 50
NP = 200
scale = .2

mask = [True] * model.EggParams.nparams

nwalkers = 800
nburnin = 500
nsample = 1000

#nwalkers = 30
#nburnin = 3
#nsample = 3

nthreads = 16 if "cosmos5" in platform.node() else 1

thetas = [model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02, mu=1.02)]
print("Thetas generated")

for i in xrange(len(thetas)):
    os.mkdir(str(i))
print("dirs created")

for i,theta in enumerate(thetas):
    #run single band chain
    sampler_s, stats_s = mcmc.run_chain(theta, nwalkers, nburnin, nsample, 
                                        nthreads=nthreads, mask=mask)
    with open(str(i) + '/single.stats.p', 'wb') as f:
        pickle.dump(stats_s,f)
    np.save(str(i) + "/single.chain.npy", sampler_s.flatchain)
    np.save(str(i) + "/single.lnprob.npy", sampler_s.flatlnprobability)

    #calculate single band priors
    weights = priors.calculate_priors(sampler_s.flatchain)
    for l, ws in weights.iteritems():
        np.save(str(i) + '/single.' + l + '.npy', ws)

    #run dual-band chain
    sampler_d, stats_d = mcmc.run_chain(theta, nwalkers, nburnin, nsample, 
                                        nthreads=nthreads, mask=mask, dual_band=True)
    with open(str(i) + '/double.stats.p', 'wb') as f:
        pickle.dump(stats_d,f)
    np.save(str(i) + "/double.chain.npy", sampler_d.flatchain)
    np.save(str(i) + "/double.lnprob.npy", sampler_d.flatlnprobability)
  
    #calculate dual-band priors
    weights = priors.calculate_priors(sampler_d.flatchain)
    for l, ws in weights.iteritems():
        np.save(str(i) + '/double.' + l + '.npy', ws)
