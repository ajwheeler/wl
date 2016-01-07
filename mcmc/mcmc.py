from __future__ import print_function
import model
import emcee
import numpy as np
import galsim
import matplotlib.pyplot as pl
import argparse

#parameter bounds
theta_lb = [0,0,-1,-1,0,0,-1,-1,-.2,-.2]
theta_ub = [20,20,1,1,20,20,1,1,.2,.2]


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

    # use g < .99 instead of g < 1 because fft can't handle g~1
    if np.sqrt(params.g1d**2 + params.g2d**2) > .9 \
       or np.sqrt(params.g1b**2 + params.g2b**2) > .9:
        return -np.inf

    try:
        gal = model.egg(params, match_image_size=data)
    except RuntimeError:
        print("error with these parameters:")
        print(params)
        return -np.inf

    diff = gal.array - data.array
    return -np.sum(diff**2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample lnprob")
    parser.add_argument('-n', '--nthreads', default=1, type=int)
    parser.add_argument('-w', '--nwalkers', default=100, type=int)
    parser.add_argument('-b', '--nburnin', default=500, type=int)
    parser.add_argument('-s', '--nsample', default=500, type=int)
    args = parser.parse_args()


    nwalkers = args.nwalkers
    ndim = 10
    theta0 = [trueParams.toArray() + 1e-4*np.random.randn(ndim) for _ in range(nwalkers)]
    #theta0 = np.random.uniform(theta_lb, theta_ub, (nwalkers,10))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data, trueParams.r_psf], threads=args.nthreads)

    nburnin = args.nburnin
    print("Burn in")
    for i, (pos, lnp, state) in enumerate(sampler.sample(theta0, iterations=nburnin)):
        if (i+1) % 100 == 0:
            print("{0:.1f}%".format(100 * float(i) / nburnin),end='')
    print()
    sampler.reset()

    nsample = args.nsample
    print("Sampling phase")
    for i, (pos, lnp, state) in enumerate(sampler.sample(pos, iterations=nsample, rstate0=state)):
        if (i+1) % 100 == 0:
            print("{0:.1f}%".format(100 * float(i) / nsample),end='')
    print()

    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
    print("Autocorrelation time:", sampler.get_autocorr_time())

    name = "%s.%s.%s" % (nwalkers, nburnin, nsample)

    f = open(name+'.stats', 'w')
    f.write("Mean acceptance fraction:"+str( np.mean(sampler.acceptance_fraction)) + "\n")
    f.write("Autocorrelation time:" + str(sampler.get_autocorr_time()))
    f.close()


    np.save(name+".chain.npy", sampler.flatchain)

    import drawcorner
    fig = drawcorner.make_figure(sampler.flatchain, trueParams.toArray())
    fig.savefig("tiny.png")
