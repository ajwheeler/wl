import numpy as np
import argparse
import mcmc


def kormendy_prior(R,F):
    """
    R: angular half-light radius of bulge
    F: total bulge flux
    """
    # Extracting kormendy from
    # http://astronomy.swin.edu.au/cosmos/K/Kormendy+Relation
    # x-axis: log (effective radius in kpc)
    # y-axis: surface brightness @ effective radius is mag arcsec^-2
    # these two points are on the line: (-0.54, 17.89), (1.56, 24.06)
    # m = a logR + b +- sigma
    a = 2.95
    b = 19.483
    sigma = 10

    # we assume that 3 arcsec ~ 6 kpc, so
    # r1/2 = 6 kpc
    # angle = 3 arcsec = 1.454e-5 rad
    distance = 6 / 1.454e-5 #kpc

    angle = R * 4.85e-6 #angular h-l R in radans
    logR = np.log10(distance * angle)

    #average surface brightness within half-light radius,
    #in normalized flux units per arcsec^2
    m0 = 126.693
    m = -2.5 * np.log10(F) + m0
    I = m / (2 * np.pi * R**2)

    I_kormendy = a * logR + b

    diff = I-I_kormendy

    return 1/np.sqrt(2*sigma**2*np.pi) * np.exp(-diff**2/2.0/sigma**2)


priors = ["kormendy", "orientation"]

def calculate_priors(chain):
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
