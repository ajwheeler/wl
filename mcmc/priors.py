import numpy as np
import argparse
import mcmc

# the Kormendy Relation
# R  = const I^-.83
# ==> R^-.54 = const I
# ==> -.54 log(R) = log(I) + const
# https://ned.ipac.caltech.edu/level5/Sept01/Kormendy/frames.html [sec 8.2]

priors = ["kormendy", "orientation"]

def calculate_priors(chain):
    sigma = .5

    power = -.54

    #R = mcmc.trueParams.rb
    #F = mcmc.trueParams.fd
    R = 1
    F = .3

    const = power*np.log(R) - np.log(F)

    weights = {}
    for l in priors:
        weights[l] = np.empty(len(chain))
    for i in xrange(len(chain)):
        #Kormendy prior
        theta = chain[i]
        R = theta[4]
        F = theta[5]

        x = power*np.log(R) - np.log(F) - const
        weights["kormendy"][i] = \
            1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x**2)/(2*sigma**2))

        #Orientation Prior
        gamma = np.sqrt(theta[2]**2  + theta[3]**2)
        if gamma > 1:
            weights["orientation"][i] = 0
        weights["orientation"][i] = 1.0/((1.0+gamma)**2)

    return weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="apply prior")
    parser.add_argument('chain_file', type=str)
    args = parser.parse_args()

    chain = np.load(args.chain_file)

    weights = calculate_priors(chain)
    prefix = args.chain_file[:-10]

    for l in priors:
        suffix = "." + l + ".npy"
        print("writing to " + prefix + suffix)
        np.save(prefix + suffix, weights[l])
        
