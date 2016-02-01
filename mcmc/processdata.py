import numpy as np
import drawcorner
import mcmc

chain = np.load("data/5000/1-31.1000.1.5000.chain.npy")

burnin_lengths = [5000,10000,300000,500000]

trueVals = mcmc.trueParams.toArray()
for l in burnin_lengths:
    subchain = chain[l:]
    np.save("data/5000/b%s.1-31.1000.1.5000.chain.npy" % l, subchain)
    fig = drawcorner.make_figure(subchain,trueVals)
    fig.savefig("data/5000/b%s.1-31.1000.1.5000.chain.npy" % l)
