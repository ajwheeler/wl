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

nwalkers = 1000
nburnin = 300
nsample = 1000

#nwalkers = 30
#nburnin = 3
#nsample = 3

nthreads = 16 if "cosmos5" in platform.node() else 1

thetas = [model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02, mu=1.02)]
for i in xrange(5):
    ub = mcmc.theta_ub
    lb = mcmc.theta_lb
    theta = [np.uniform(low=l, high=u) for l,u n zip(lb,ub)]
    thetas.append(theta)
print("Thetas generated")
print(thetas)

for i in xrange(len(thetas)):
    os.mkdir(str(i))
print("dirs created")

for i,theta in enumerate(thetas):
    mag_mask = [True]*11
    mag_mask[-2] = False
    mag_mask[-3] = False

    shear_mask = [True]*10 + [False]
    for l,m in [("shear", shear_mask), ("mag", mag_mask)]:
        #run single band chain
        sampler_s, stats_s = mcmc.run_chain(theta, nwalkers, nburnin, nsample, 
                                            nthreads=nthreads, mask=m)
        with open(str(i) + '/' + l + '.single.stats.p', 'wb') as f:
            pickle.dump(stats_s,f)
        np.save(str(i) + '/' + l + '.single.chain.npy', sampler_s.flatchain)
        np.save(str(i) + '/' + l + '.single.lnprob.npy', sampler_s.flatlnprobability)

        #calculate single band priors
        weights = priors.calculate_priors(sampler_s.flatchain)
        for p, ws in weights.iteritems():
            np.save(str(i) + '/' + l + '.single.' + p + '.npy', ws)

        #run dual-band chain
        sampler_d, stats_d = mcmc.run_chain(theta, nwalkers, nburnin, nsample, 
                                            nthreads=nthreads, mask=m, dual_band=True)
        with open(str(i) + '/' + l + '.double.stats.p', 'wb') as f:
            pickle.dump(stats_d,f)
        np.save(str(i) + '/' + l + '.double.chain.npy', sampler_d.flatchain)
        np.save(str(i) + '/' + l + '.double.lnprob.npy', sampler_d.flatlnprobability)

        #calculate dual-band priors
        weights = priors.calculate_priors(sampler_d.flatchain)
        for p, ws in weights.iteritems():
            np.save(str(i) + '/' + l + '.double.' + p + '.npy', ws)
