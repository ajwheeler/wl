import corner
from itertools import compress
import sys

def make_figure(samples, true_vals, weights=None, mask=[True]*10):
    labels = [r"$R_{disk}$", r"$F_{disk}$", r"$\gamma_1^{disk}$", r"$\gamma_2^{disk}$",
              r"$R_{bulge}$", r"$F_{bulge}$", r"$\gamma_1^{bulge}$", r"$\gamma_2^{bulge}$",
              r"$\gamma_1^{shear}$", r"$\gamma_2^{shear}$"]
    labels = list(compress(labels, mask))
    ranges = [(0,8), (0,5), (-1,1),(-1,1),(0,7), (0,4),(-1,1),(-1,1),(-.2,.2),(-.2,.2)]
    figure = corner.corner(samples, labels=labels,
                           truths=true_vals,
                           weights=weights,
                           show_titles=True,
                           range=ranges)
    
    
    return figure

if __name__ == '__main__':
    import numpy as np
    import model

    inputf = sys.argv[1]
    outputf = sys.argv[2]

    weightfiles = sys.argv[3:]
    if weightfiles != []:
        print(weightfiles)
        weights = np.prod(np.array([np.load(f) for f in weightfiles]), axis=0)
    else:
        weights = None

    chain = np.load(inputf)
    trueParams = model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02)
    fig = make_figure(chain, trueParams.toArray(), weights=weights)

    print("Wrinting output to " + outputf)
    fig.savefig(outputf)
