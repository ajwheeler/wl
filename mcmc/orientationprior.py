import numpy as np
import argparse

parser = argparse.ArgumentParser(description="apply prior")
parser.add_argument('chain_file', type=str)
parser.add_argument('output_file', type=str)
args = parser.parse_args()

chain = np.load(args.chain_file)

weights = []
for theta in chain:
    gamma = np.sqrt(theta[2]**2  + theta[3]**2)
    if gamma > 1:
        weights.append(0)
    weights.append(gamma)

weights = np.array(weights)
np.save(args.output_file, weights)
