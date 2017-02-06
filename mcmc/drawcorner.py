import matplotlib
matplotlib.use('Agg')
import corner
from itertools import compress
import sys
import model
import mcmc
import pickle

def make_figure(samples, true_vals, weights=None, stats=None,
                mask=[True]*model.EggParams.nparams, enforce_ranges=False):
    labels = [r"$R_{disk}$", r"$F_{disk}$", r"$g_1^{disk}$", r"$g_2^{disk}$",
              r"$R_{bulge}$", r"$F_{bulge}$", r"$g_1^{bulge}$", r"$g_2^{bulge}$",
              r"$g_1^{shear}$", r"$g_2^{shear}$", r"$\mu$"]
    labels = ["(%s) " % v + l for (l,v) in zip(compress(labels,mask), true_vals)]

    if enforce_ranges:
        ranges = [(0,8), (0,5), (-1,1),(-1,1),(0,7), (0,4),(-1,1),(-1,1),(-.2,.2),(-.2,.2), (.8,1.2)]
        ranges = list(compress(ranges, mask))
    else:
        ranges = None

    figure = corner.corner(samples, labels=labels,
                           truths=true_vals,
                           weights=weights,
                           show_titles=True,
                           range=ranges)
    if stats:
        notes = ""
        for k,v in stats.iteritems():
            notes += "%s: %s\n" % (k,v)

    figure.text(.6, .7, notes)
    return figure

if __name__ == '__main__':
    import numpy as np
    import model

    inputf = sys.argv[1]
    statsf = sys.argv[2]
    outputf = sys.argv[3]

    weightfiles = sys.argv[4:]
    if weightfiles != []:
        print(weightfiles)
        weights = np.prod(np.array([np.load(f) for f in weightfiles]), axis=0)
    else:
        weights = None

    chain = np.load(inputf)

    with open(statsf) as f:
        stats = pickle.load(f)

    #TODO: get from stats?
    trueTheta = model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02, mu=1.02).toArray()

    mask = [True, True, True, True, True, True, True, True, True, True, True]

    fig = make_figure(chain, trueTheta, stats=stats, weights=weights, mask=mask)

    print("Wrinting output to " + outputf)
    fig.savefig(outputf, bbox_inches='tight')
