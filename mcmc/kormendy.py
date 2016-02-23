import numpy as np
import argparse
import mcmc

# the Kormendy Relation
# R  = const I^-.83
# ==> R^-.54 = const I
# ==> -.54 log(R) = log(I) + const
# https://ned.ipac.caltech.edu/level5/Sept01/Kormendy/frames.html [sec 8.2]

sigma = .5

power = -.54

R = mcmc.trueParams.rb
F = mcmc.trueParams.fd
const = power*np.log(R) - np.log(F)

parser = argparse.ArgumentParser(description="apply prior")
parser.add_argument('chain_file', type=str)
parser.add_argument('output_file', type=str)
args = parser.parse_args()

chain = np.load(args.chain_file)
weights = np.empty(len(chain))
for i in xrange(len(chain)):
    theta = chain[i]
    R = theta[4]
    F = theta[5]
    
    x = power*np.log(R) - np.log(F) - const
    weights[i] = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x**2)/(2*sigma**2))
    
np.save(args.output_file, weights)
