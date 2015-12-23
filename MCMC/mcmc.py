import model
import emcee
import numpy as np
import galsim
import matplotlib.pyplot as pl

theta_lb = [0,0,-1,-1,0,0,-1,-1,-1,-1]
theta_ub = [20,20,1,1,20,20,1,1,1,1]

class QuietImage(galsim.image.Image):
    """This is a hack so that the error output if emcee has an error calling
    lnprob the output will not be insanely long"""
    def __repr__(self):
        return "<galsim image with %s>" % self.bounds

    def __str__(self):
        return "<galsim image with %s>" % self.bounds

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

nwalkers = 100
ndim = 10
#theta0 = [model.EggParams().toArray() + 1e-4*np.random.randn(ndim) for _ in range(nwalkers)]
theta0 = np.random.uniform(theta_lb, theta_ub, (nwalkers,10))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data, trueParams.r_psf])


pos, prob, state = sampler.run_mcmc(theta0, 100)
sampler.reset()
sampler.run_mcmc(pos, 1000, rstate0=state)
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
print("Autocorrelation time:", sampler.get_autocorr_time())

import drawcorner
fig = drawcorner.make_figure(sampler.flatchain, trueParams.toArray())
fig.savefig("uniformtheata0.png")
