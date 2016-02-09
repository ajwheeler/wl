import numpy as np
import argparse

parser = argparse.ArgumentParser(description="apply prior")
parser.add_argument('chain_file', type=str)
parser.add_argument('output_file', type=str)
args = parser.parse_args()

chain = np.load(args.chain_file)

accepted = []
for theta in chain:
    gamma = np.sqrt(theta[2]**2  + theta[3]**2)
    if gamma > 1:
        pass
    if np.random.rand() < gamma:
        accepted.append(theta)

print(str(len(accepted)) + " accepted out of " + str(len(chain)))
np.save(args.output_file, np.array(accepted))
