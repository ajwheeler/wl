import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner
from itertools import compress
import sys
import model
import mcmc
import pickle
import numpy as np
import argparse

labels = [r"$R_{disk}$", r"$F_{disk}$", r"$g_1^{disk}$", r"$g_2^{disk}$",
          r"$R_{bulge}$", r"$F_{bulge}$", r"$g_1^{bulge}$", r"$g_2^{bulge}$",
          r"$g_1^{shear}$", r"$g_2^{shear}$", r"$\mu$"]

def make_figure(samples, true_vals, weights=None, stats=None,
                mask=[True]*model.EggParams.nparams, enforce_ranges=False):
    if enforce_ranges:
        ranges = [(0,8), (0,5), (-1,1),(-1,1),(0,7), (0,4),(-1,1),(-1,1),
                  (-.2,.2),(-.2,.2), (.8,1.2)]
        ranges = list(compress(ranges, mask))
    else:
        ranges = None

    ls = ["(%s) " % v + l for (l,v) in zip(compress(labels,mask), true_vals)]

    figure = corner.corner(samples, labels=ls, truths=true_vals,
                           weights=weights, show_titles=True, range=ranges)
    if stats:
        notes = ""
        for k,v in stats.iteritems():
            if k != 'time' and k != 'true_params':
                notes += "%s: %s\n" % (k,v)
        figure.text(.6, .7, notes)

    return figure

def make_trace(samples, stats) :
    mask = stats['mask']
    truths = stats['true_params'].toArray(mask=mask)
    ls = list(compress(labels,mask))

    xs = range(chain.shape[0])

    nparams = sum(mask)
    fig = plt.figure(figsize=(10,3*nparams))
    for i in xrange(nparams):
        plt.subplot(nparams,1,i)
        plt.plot(xs,chain[:,i])
        plt.ylabel(ls[i])
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="draw plots from chains")
    parser.add_argument("-s", '--statsfile', type=str)
    parser.add_argument("-c", '--chainfile', type=str)
    parser.add_argument("-o", '--output', type=str)
    parser.add_argument("-t", '--trace', action='store_true')
    args = parser.parse_args()

#    if weightfiles != []:
#        print(weightfiles)
#        weights = np.prod(np.array([np.load(f) for f in weightfiles]), axis=0)
#    else:
    weights = None

    chain = np.load(args.chainfile)

    with open(args.statsfile) as f:
        stats = pickle.load(f)

    if args.trace:
        fig = make_trace(chain, stats)
    else:
        truths = stats['true_params'].toArray()
        fig = make_figure(chain, truths, stats=stats, weights=weights,
                          mask=stats['mask'])


    print("Wrinting output to " + args.output)
    fig.savefig(args.output, bbox_inches='tight')
