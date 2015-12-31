from __future__ import print_function
import model
import emcee
import numpy as np
import galsim
import matplotlib.pyplot as pl
import argparse

#parameter bounds
theta_lb = [0,0,-1,-1,0,0,-1,-1,-1,-1]
theta_ub = [20,20,1,1,20,20,1,1,1,1]


class QuietImage(galsim.image.Image):
    """This is a hack so that the error output if emcee has an error calling
    lnprob the output will not be insanely long"""
    def __repr__(self):
        return "<galsim image with %s>" % self.bounds

    def __str__(self):
        return "<galsim image with %s>" % self.bounds


parser = argparse.ArgumentParser(description="Sample lnprob")
parser.add_argument('-n', '--nthreads', default=1, type=int)
args = parser.parse_args()

trueParams = model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02)
data = model.egg(trueParams)
data.__class__ = QuietImage

def lnprob(theta, data, r_psf):
    if not all(theta > theta_lb) or not all(theta < theta_ub):
        return -np.inf

    params = model.EggParams(r_psf=r_psf)
    params.fromArray(theta)
    gal = model.egg(params, match_image_size=data)
    diff = gal.array - data.array
    return -np.sum(diff**2)

trueParams = model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02)
#trueParams = model.EggParams()

nwalkers = 1000
ndim = 10
theta0 = [trueParams.toArray() + 1e-4*np.random.randn(ndim) for _ in range(nwalkers)]
#theta0 = np.random.uniform(theta_lb, theta_ub, (nwalkers,10))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data, trueParams.r_psf], threads=args.nthreads)

nburnin = 1000
print("Burn in")
for i, (pos, lnp, state) in enumerate(sampler.sample(theta0, iterations=nburnin)):
    if (i+1) % 100 == 0:
        print("{0:.1f}%".format(100 * float(i) / nburnin),end='')
print()
sampler.reset()

nsample = 5000
print("Sampling phase")
for i, (pos, lnp, state) in enumerate(sampler.sample(pos, iterations=nsample, rstate0=state)):
    if (i+1) % 100 == 0:
        print("{0:.1f}%".format(100 * float(i) / nsample),end='')
print()

print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
print("Autocorrelation time:", sampler.get_autocorr_time())

import drawcorner
fig = drawcorner.make_figure(sampler.flatchain, trueParams.toArray())
fig.savefig("tiny.png")


f = open('stats', 'w')
f.write("Mean acceptance fraction:"+str( np.mean(sampler.acceptance_fraction)))
f.write("Autocorrelation time:" + str(sampler.get_autocorr_time()))
f.close()


np.save("chain.npy", sampler.flatchain)

