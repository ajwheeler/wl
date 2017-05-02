import numpy as np
import corner

def median(chain, weights=None):
    """calculate mode and errorbars from chain"""
    _, l = chain.shape

    if not weights:
        return np.array([corner.quantile(chain[:,i], [0.16, 0.5, 0.84]) for i in range(l)])
    else:
        return np.array([corner.quantile(chain[:,i], [0.16, 0.5, 0.84], weights=weights[:,i]) for i in range(l)])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate mode and error bars"+\
                    " from chain and weightfiles")
    parser.add_argument("chain_file")
    parser.add_argument("weight_file", nargs="*")
    args = parser.parse_args()

    if args.weight_file:
        weights = np.prod(np.array([np.load(f) for f in args.weight_file]), 0)
    else:
        weights = None

    chain = np.load(args.chain_file)
    stats = median(chain, weights=weights)

    print(stats)
