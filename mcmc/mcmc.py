#!/usr/bin/env python
from  __future__ import print_function
import model
import emcee
import numpy as np
import galsim
import argparse
import datetime
import copy
import pickle
import itertools

class QuietImage(galsim.image.Image):
    """This is a hack so that the error output if emcee has an error calling
    lnprob the output will not be insanely long"""
    def __repr__(self):
        return "<galsim image with %s>" % self.bounds

    def __str__(self):
        return "<galsim image with %s>" % self.bounds


#parameter bounds
theta_lb = [0.,0.,-1.,-1.,0.,0.,-1.,-1.,-.1,-.1, 0.8]
theta_ub = [8.,5., 1., 1.,7.,4., 1., 1., .1, .1, 1.2]

def FlatPrior(cube, ndim, nparams):
    """turn the unit cube into the parameter cube, as per pymultinest"""
    for i in xrange(ndim):
        lb = theta_lb[i]
        ub = theta_ub[i]
        cube[i] = cube[i]*(ub-lb) - lb

#TODO: move to model.py?
def within_bounds(theta, mask):
    #remove bounds for fixed parameters
    lb = list(itertools.compress(theta_lb, mask))
    ub = list(itertools.compress(theta_ub, mask))
    return  all(theta > lb) and all(theta < ub)

def lnprob(theta, data, dual_band, summed, pixel_var, psf, mask, trueParams):
    """log likelyhood for emcee"""
    theta = np.array(theta)

    #TODO: use within_bounds
    #remove bounds for fixed parameters
    lb = list(itertools.compress(theta_lb, mask))
    ub = list(itertools.compress(theta_ub, mask))
    if not all(theta > lb) or not all(theta < ub):
        return -np.inf

    params = copy.copy(trueParams)
    params.fromArray(theta, mask)

    #require that disk is larger than bulge
    if params.rd < params.rb:
        return -np.inf

    #use g < .9 instead of g < 1 because FFT can't handle g close to 1
    if np.sqrt(params.g1d**2 + params.g2d**2) > .9 \
       or np.sqrt(params.g1b**2 + params.g2b**2) > .9:
        return -np.inf

    #a single image (i.e. not a pair of g/r band images)
    #for the model to size-match
    single_image = data[0] if dual_band else data

    try:
        gals = model.egg(params, match_image_size=single_image,
                         dual_band=dual_band, r_psf=psf)
    except RuntimeError:
        print("error drawing galaxy with these parameters:")
        print(params)
        return -np.inf

    if summed:
        diff = (gals[0]+gals[1]).array - (data[0] + data[1]).array
        p = -np.sum(diff**2)
    elif dual_band:
        g_diff = gals[0].array - data[0].array
        r_diff = gals[1].array - data[1].array
        p = -np.sum(g_diff**2) - np.sum(r_diff**2)
    else:
        diff = gals.array - data.array
        p = -np.sum(diff**2)

    return p * .5/pixel_var

def generate_data(trueParams, dual_band, NP=200, SNR=50, psf=.25):
    data = model.egg(trueParams, dual_band=dual_band, nx=NP, ny=NP, r_psf=psf)

    #apply noise and make data a QuietImage (see class at top of file)
    if dual_band:
        combined = data[0] + data[1]
        var = combined.addNoiseSNR(galsim.GaussianNoise(), SNR,
                                   preserve_flux=True)
        for i in [0,1]:
            seed = long(datetime.datetime.now().microsecond)
            noise = galsim.GaussianNoise(galsim.BaseDeviate(seed))
            noise = noise.withVariance(var/2)
            data[i].addNoise(noise)

        data[0].__class__ = QuietImage #g band image
        data[1].__class__ = QuietImage #r band image
    else:
        var = data.addNoiseSNR(galsim.GaussianNoise(),SNR,preserve_flux=True)
        data.__class__ = QuietImage

    return data, var

def run_chain(data, pixel_var, trueParams, nwalkers, nburnin, nsample, nthreads=1,
              mask=True*model.EggParams.nparams, parallel_tempered=False,
              dual_band=False, summed=False, NP=200, SNR=50, psf=0.25):

    t1 = datetime.datetime.now()
    ndim = mask.count(True)
    if parallel_tempered:
        ntemps = 20

        theta0 = [[model.EggParams().toArray(mask) + 1e-2*np.random.randn(ndim) \
                   for _ in range(nwalkers)] for _ in range(ntemps)]
        #flat prior
        def logp(x):
            return 0.01
        sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnprob, logp,
                                  loglargs=[data, dual_band, summed, pixel_var,
                                            psf, mask, trueParams],
                                  threads=nthreads)
    else:
        theta0 = [model.EggParams().toArray(mask) + 1e-2*np.random.randn(ndim)\
                  for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=[data, dual_band, summed,
                                              pixel_var, psf, mask, trueParams],
                                        threads=nthreads)

    pos, _, state = sampler.run_mcmc(theta0, nburnin)
    sampler.reset()
    sampler.run_mcmc(pos, nsample, rstate0=state)

    t2 = datetime.datetime.now()

    #collect stats
    stats = {}
    stats['acceptance_fraction'] = np.mean(sampler.acceptance_fraction)
    stats['autocorrelation_time'] = sampler.get_autocorr_time()
    stats['true_params'] = trueParams
    stats["mask"] = mask
    stats['nburnin'] = nburnin
    stats['nwalkers'] = nwalkers
    stats['nsample'] = nsample
    stats['nthreads'] = nthreads
    stats['parallel_tempered'] = parallel_tempered
    stats['dual_band'] = dual_band
    stats['NP'] = NP
    stats['SNR'] = SNR
    stats['time'] = datetime.datetime.now()
    stats['computetime'] = t2 - t1

    return sampler, stats


#     #    #    ### #     #
##   ##   # #    #  ##    #
# # # #  #   #   #  # #   #
#  #  # #     #  #  #  #  #
#     # #     #  #  #  #  #
#     # #######  #  #   # #
#     # #     # ### #    ##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample lnprob")
    parser.add_argument('-n', '--nthreads', default=1, type=int)
    parser.add_argument('-w', '--nwalkers', default=30, type=int)
    parser.add_argument('-b', '--nburnin', default=10, type=int)
    parser.add_argument('-s', '--nsample', default=10, type=int)
    parser.add_argument('-p', '--parallel-tempered', action='store_true')
    parser.add_argument('-d', '--draw-plot', action='store_true')
    parser.add_argument('-2', '--dual-band', action='store_true')
    parser.add_argument('--summed', action='store_true')
    parser.add_argument('--snr', default=50, type=int)
    parser.add_argument('--suffix', default=None, type=str)
    parser.add_argument('--mask', default='nomag', type=str)
    parser.add_argument('--sampler', default='emcee', type=str,
                        choices=['emcee', 'multinest', 'gridsampler', 'none'])
    parser.add_argument('--loaddata', type=str, default=None)
    parser.add_argument('--drawdata', type=str, default=None)
    args = parser.parse_args()

    NP = 200 #maybe this will become a cli arg?

    assert((not args.summed) or args.dual_band)

    #set mask -- which params to fit
    if args.mask == 'nolensing':
        # no lensing params (8)
        mask = [True]*8 + [False]*3
    elif args.mask == 'justdisk':
        #just the disk params (4)
        mask = [True]*4 + [False]*7
    elif args.mask == 'nomag':
        #by default, do everything except magnification (10)
        mask = [True]*11
        mask[-1] = False
    elif args.mask == 'all':
        #fit all params (11)
        mask = [True]*11
    else:
        #custom mask
        assert(len(args.mask) == 11)
        mask = [True if i == '1' else False for i in args.mask]
    print("mask = " + str([1 if m else 0 for m in mask]))

    #construct name
    name = "%s.%s.%s" % (args.nwalkers, args.nburnin, args.nsample)
    if args.parallel_tempered:
        name += '.pt'
    if args.summed:
        name += '.summed'
    elif args.dual_band:
        name += '.dual'
    else:
        name += '.single'
    if args.suffix != None:
        name += "." + args.suffix
    t = datetime.datetime.now()
    name = str(t.month) + "-" + str(t.day) + "." + name

    #generate or load data
    if args.loaddata:
        with open(args.loaddata,'r') as f:
            trueParams, pixel_var, args.snr, data = pickle.load(f)
        if type(data) == tuple and not args.dual_band:
            print("converting dual-band image to single band")
            data = data[0] + data[1]
        elif type(data) != tuple and args.dual_band:
            raise RuntimeError("single band data incompatible with dual-band fit")
    else: #generate data
        #true params, simulated data
        trueParams = model.EggParams(g1d=.2, g2d=.3, g2b=.4, g1s=0,
                                     g2s = 0, mu=1)
        data, pixel_var = generate_data(trueParams, args.dual_band, NP, args.snr)

        datafilename = name + '.p'
        with open(datafilename,'w') as f:
            pickle.dump((trueParams, pixel_var, args.snr, data), f)

    #save image of simulated data if requested
    if args.drawdata:
        #import matplotlib
        #matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.imshow(data.array, cmap=plt.get_cmap('gray'))
        plt.savefig(args.drawdata)


    #sample with specified sampler

    if args.sampler == 'gridsampler':
        #use a simple grid sampler instead of MCMC
        grid = []
        grid.append(np.linspace(2.9,3,5,60))
        grid.append(np.linspace(0.6,.9,20))
        #grid.append(np.linspace(0,1,100))
        #grid.append(np.linspace(0,1,100))
        grid = grid + [None]*9 #TODO pick grid spacing for other params

        print('grid')
        points = list(itertools.product(*itertools.compress(grid,mask)))
        print('points')
        logls = [lnprob(p, data, args.dual_band, args.summed, pixel_var, mask, trueParams)\
                 for p in points]
        print('logls')
        i = np.argmax(logls)
        print('max')
        print("best params: " + str(points[i]))

    elif args.sampler == 'multinest':
        import pymultinest
        ndim = mask.count(True)

        #define the log likelihood for multinest
        def loglikelyhood(cube, ndim, nparams, lnew):
            return lnprob(cube, data, args.dual_band, args,summed, pixel_var, mask, trueParams)

        pymultinest.run(loglikelyhood, FlatPrior, ndim,n_live_points=100,
                        multimodal=False)


    elif args.sampler == 'emcee':
        sampler, stats = run_chain(data, pixel_var, trueParams, args.nwalkers,
                                   args.nburnin, args.nsample, args.nthreads, mask,
                                   args.parallel_tempered, NP=NP,
                                   dual_band=args.dual_band, summed=args.summed,
                                   SNR=args.snr)
        print()
        print("chain finished!")
        print()
        stats['data_image'] = args.loaddata if args.loaddata else datafilename
        print(stats)

        #write stats
        with open(name+'.stats.p', 'wb') as f:
            pickle.dump(stats,f)

        #save chain/lnprobs
        np.save(name+".chain.npy", sampler.flatchain)
        np.save(name+".lnprob.npy", sampler.flatlnprobability)

        #draw corner plot
        if args.draw_plot:
            import drawcorner
            chain = sampler.flatchain
            if args.parallel_tempered:
                chain = chain.reshape(ntemps*args.nwalkers*args.nsample, ndim)
            fig = drawcorner.make_figure(chain, trueParams.toArray(mask),
                                         mask=mask, stats=stats)
            print("writing plot to " + name + ".png")
            fig.savefig(name + ".png", bbox_inches='tight')
