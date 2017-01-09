from  __future__ import print_function
import model
import emcee
import numpy as np
import galsim
import argparse
import time
import copy
import pickle
import sys
import itertools 

class QuietImage(galsim.image.Image):
    """This is a hack so that the error output if emcee has an error calling
    lnprob the output will not be insanely long"""
    def __repr__(self):
        return "<galsim image with %s>" % self.bounds

    def __str__(self):
        return "<galsim image with %s>" % self.bounds


#parameter bounds
theta_lb = [0,0,-1,-1,0,0,-1,-1,-.1,-.1, 0.8]
theta_ub = [8,5, 1, 1,7,4, 1, 1, .1, .1, 1.2]

def lnprob(theta, data, dual_band, pixel_var, mask, trueParams):
    theta = np.array(theta)

    #remove bounds for fixed parameters
    lb = list(itertools.compress(theta_lb, mask))
    ub = list(itertools.compress(theta_ub, mask))
    if not all(theta > lb) or not all(theta < ub):
        return -np.inf

    params = model.EggParams()
    params = copy.copy(trueParams)
    params.fromArray(theta, mask)

    # use g < .9 instead of g < 1 because fft can't handle g~1
    if np.sqrt(params.g1d**2 + params.g2d**2) > .9 \
       or np.sqrt(params.g1b**2 + params.g2b**2) > .9:
        return -np.inf

    try:
        #a single imgage (i.e. not a pair of g/r band images
        #for the model to size-match
        single_image = data[0] if dual_band else data
        gals = model.egg(params, match_image_size=single_image, dual_band=dual_band)
    except RuntimeError:
        print("error drawing galaxy with these parameters:")
        print(params)
        return -np.inf

    if dual_band:
        g_diff = gals[0].array - data[0].array
        r_diff = gals[1].array - data[1].array
        p = -(np.sum(g_diff**2) + np.sum(r_diff**2))
    else:
        diff = gals.array - data.array
        p = -np.sum(diff**2)

    return p * .5/pixel_var

def generate_data(trueParams, dual_band=False, NP=200, scale=.2, SNR=50):
    data = model.egg(trueParams, dual_band=dual_band, nx=NP, ny=NP, scale=scale)
    
    #apply noise and make data a QuietImage (see class at top of file)
    if dual_band:
        for i in [0,1]:
            #bd = galsim.BaseDeviate(int(time.time()))
            data[i].addNoiseSNR(galsim.GaussianNoise(),SNR,preserve_flux=True)

        print("WARNING: SNR may be incorrect")
        
        data[0].__class__ = QuietImage #g band image
        data[1].__class__ = QuietImage #r band image
    else:
        #bd = galsim.BaseDeviate(int(time.time()))
        #data.addNoise(galsim.GaussianNoise(bd, pixel_noise))
        data.addNoiseSNR(galsim.GaussianNoise(),SNR,preserve_flux=True)
        data.__class__ = QuietImage
    
    return data

def run_chain(trueParams, nwalkers, nburnin, nsample, nthreads=1,
              mask=True*model.EggParams.nparams, parallel_tempered=False,
              dual_band=False, NP=200, scale=.2, SNR=50):

    data = generate_data(trueParams, dual_band, NP, scale, SNR)
    pixel_noise = (scale)**2/(np.pi * trueParams.rd**2 * SNR)

    # print('fix this!')
    # import matplotlib
    # matplotlib.use('Agg') 
    # import matplotlib.pyplot as plt
    # plt.imshow(data.array, cmap=plt.get_cmap('gray'))
    # plt.savefig("out.png")

    ndim = mask.count(True)
    if parallel_tempered:
        ntemps = 20
        theta0 = [[trueParams.toArray(mask) + 1e-2*np.random.randn(ndim) \
                   for _ in range(nwalkers)] for _ in range(ntemps)]
        #flat prior
        def logp(x):
            return 0.01
        sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnprob, logp, 
                                  loglargs=[data, dual_band, pixel_noise**2, mask, trueParams],
                                  threads=nthreads)
    else:
        theta0 = [trueParams.toArray(mask) + 1e-2*np.random.randn(ndim)\
                  for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                        args=[data, dual_band, pixel_noise**2, mask, trueParams], 
                                        threads=nthreads)

    pos, _, state = sampler.run_mcmc(theta0, nburnin)
    sampler.reset()
    sampler.run_mcmc(pos, nsample, rstate0=state)

    #collect stats
    stats = {}
    stats['acceptance_fraction'] = np.mean(sampler.acceptance_fraction)
    stats['autocorrelation_time'] = sampler.get_autocorr_time()
    stats['fiducial_params'] = trueParams
    stats["mask"] = mask
    stats['nburnin'] = nburnin
    stats['nwalkers'] = nwalkers
    stats['nsample'] = nsample
    stats['nthreads'] = nthreads
    stats['parallel_tempered'] = parallel_tempered
    stats['dual_band'] = dual_band
    stats['NP'] = NP
    stats['SNR'] = SNR
    stats['scale'] = scale
    stats['time'] = time.localtime()

    return sampler, stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample lnprob")
    parser.add_argument('-n', '--nthreads', default=1, type=int)
    parser.add_argument('-w', '--nwalkers', default=30, type=int)
    parser.add_argument('-b', '--nburnin', default=10, type=int)
    parser.add_argument('-s', '--nsample', default=10, type=int)
    parser.add_argument('-p', '--parallel-tempered', action='store_true')
    parser.add_argument('-d', '--draw-plot', action='store_true')
    parser.add_argument('-2', '--dual-band', action='store_true')
    parser.add_argument('--nolensing', action='store_true')
    parser.add_argument('--justdisk', action='store_true')
    parser.add_argument('--snr', default=50, type=int)
    parser.add_argument('--suffix', default=None, type=str)
    parser.add_argument('--mask', default=None, type=str)
    parser.add_argument('--gridsampler', action='store_true')
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, "=", getattr(args, arg))
    
    NP = 200
    SCALE = .2

    #set mask -- which params to fit
    if args.nolensing:
        mask = [True]*8 + [False]*3
    elif args.justdisk:
        mask = [True]*4 + [False]*7
    elif args.mask:
        assert(len(args.mask) == 11)
        mask = [True if i == '1' else False for i in args.mask]
    else:
        #by default, do everything except magnification
        mask = [True]*model.EggParams.nparams
        mask[-1] = False

    print("mask = " + str(mask))

    #true params
    trueParams = model.EggParams(g1d=.2, g2d=.3, g2b=.4, g1s=.01,
                                 g2s = .02, mu=1.02)

    if args.gridsampler:
        data = generate_data(trueParams, args.dual_band, NP, SCALE, args.snr)
        pixel_noise = (SCALE)**2/(np.pi * trueParams.rd**2 * args.snr)

        #use a simple grid sampler instead of MCMC
        grid = []
        grid.append(np.linspace(2,4,200))
        grid.append(np.linspace(0,1,100))
        grid.append(np.linspace(0,1,100))
        grid.append(np.linspace(0,1,100))
        grid = grid + [None]*7 #TODO pick grid spacing for other params

        points = list(itertools.product(*itertools.compress(grid,mask)))
        logls = [lnprob(p, data, args.dual_band, pixel_noise**2, mask, trueParams)\
                 for p in points]

        i = np.argmax(logls)
        print("best params: " + str(points[i]))
        sys.exit()

    sampler, stats = run_chain(trueParams, args.nwalkers,
                               args.nburnin, args.nsample,
                               args.nthreads, mask,
                               args.parallel_tempered, NP=NP,
                               scale=SCALE, dual_band=args.dual_band,
                               SNR=args.snr)
    print()
    print("chain finished!")
    print()
    print(stats)

    #construct name
    name = "%s.%s.%s" % (args.nwalkers, args.nburnin, args.nsample)
    if args.parallel_tempered:
        name += '.pt'
    name += ".dual" if args.dual_band else ".single"
    if args.suffix != None:
        name += "." + args.suffix
    t = time.localtime()
    name = str(t.tm_mon) + "-" + str(t.tm_mday) + "." + name

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
        fig = drawcorner.make_figure(chain, trueParams.toArray(mask), mask=mask)
        print("writing plot to " + name + ".png")
        fig.savefig(name + ".png")
