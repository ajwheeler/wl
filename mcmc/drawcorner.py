import corner
from itertools import compress
import sys

def make_figure(samples, true_vals, mask=[True]*10):
    labels = [r"$R_{disk}$", r"$F_{disk}$", r"$\gamma_1^{disk}$", r"$\gamma_2^{disk}$",
              r"$R_{bulge}$", r"$F_{bulge}$", r"$\gamma_1^{bulge}$", r"$\gamma_2^{bulge}$",
              r"$\gamma_1^{shear}$", r"$\gamma_2^{shear}$"]
    labels = list(compress(labels, mask))
    figure = corner.corner(samples, labels=labels,
                           truths=true_vals,
                           show_titles=True)
    
    
    return figure

if __name__ == '__main__':
    import numpy as np
    import model

    inputf = sys.argv[1]
    outputf = inputf[:-9] + "png" # npy -> png

    chain = np.load("500.500.1500.1-15.chain.sampled.npy")
    trueParams = model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02)
    fig = make_figure(chain, trueParams.toArray())

    print("Wrinting output to " + outputf)
    fig.savefig(outputf)
