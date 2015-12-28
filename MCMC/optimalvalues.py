import numpy as np

def optimal_values(sampler, N):
    sprobs = sorted(sampler.flatlnprobability)
    thetas = []

    i = 0
    while len(thetas) < N:
        ts = np.unique(sampler.flatchain[np.where(sampler.flatlnprobability == sprobs[i])])
        thetas.append(ts)
        i += 1

    return thetas
