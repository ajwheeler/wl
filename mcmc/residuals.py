import argparse
import pickle
import matplotlib.pyplot as plt
from mcmc import QuietImage
import model
import numpy as np

def show_residual(im1, im2):
    diff = im1-im2
    plt.imshow(diff.array)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='draw gal image from pickled file')
    parser.add_argument('data_file')
    parser.add_argument('stats_file')
    args = parser.parse_args()


    with open(args.data_file,'r') as f:
        data = pickle.load(f)[-1]

    with open(args.stats_file, 'r') as f:
        stats = pickle.load(f)

    params = [4.46, 0.53, 0.26, 0.29, 2.34, .73, .03, 0.41]
    theta = model.EggParams()
    theta.fromArray(np.array(params), mask=stats["mask"])
    trueTheta = stats["true_params"]

    bestGuess = model.egg(theta, match_image_size=data)
    noiseless = model.egg(trueTheta, match_image_size=data)

    show_residual(data, bestGuess)
    show_residual(data, noiseless)
    show_residual(noiseless, bestGuess)
