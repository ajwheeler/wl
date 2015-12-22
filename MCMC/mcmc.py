import model
import emcee
import numpy as np
import galsim
import matplotlib.pyplot as pl

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
    #return -inf if theta is outside of allowed values
    max_r = 50
    if not all(theta > [0,0,-1,-1,0,0,-1,-1,-1,-1])\
       or not all(theta < [max_r,np.inf,1,1,max_r,np.inf,1,1,1,1]):
        return -np.inf

    params = model.EggParams(r_psf=r_psf)
    params.fromArray(theta)
    gal = model.egg(params, match_image_size=data)
    diff = gal.array - data.array
    return np.sum(diff**2)

trueParams = model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02)

nwalkers = 22
ndim = 10
theta0 = [model.EggParams().toArray() + 1e-4*np.random.randn(ndim) for _ in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data, trueParams.r_psf])


pos, prob, state = sampler.run_mcmc(theta0, 40)
sampler.reset()
sampler.run_mcmc(pos, 100, rstate0=state)
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
print("Autocorrelation time:", sampler.get_autocorr_time())

for i in range(ndim):
    pl.figure()
    pl.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
    pl.title("Dimension {0:d}".format(i))

pl.show()
