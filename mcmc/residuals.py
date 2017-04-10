import argparse
import pickle
import matplotlib.pyplot as plt
from mcmc import QuietImage
import model
import numpy as np

def show_residual(im1, im2):
    if type(im1) == tuple:
        plt.subplot(121)
        plt.imshow((im1[0]-im2[0]).array)
        plt.colorbar()

        plt.subplot(122)
        plt.imshow((im1[1]-im2[1]).array)
        plt.colorbar()
    else:
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

    params = [3.36, 0.34, 0.25, 0.23, 3.11, .92, .05, 0.36]
    theta = model.EggParams()
    theta.fromArray(np.array(params), mask=stats["mask"])
    trueTheta = stats["true_params"]


    bestGuess = model.egg(theta, match_image_size=data,
                          dual_band=stats["dual_band"])
    noiseless = model.egg(trueTheta, match_image_size=data,
                          dual_band=stats["dual_band"])

    show_residual(data, bestGuess)
    show_residual(data, noiseless)
    show_residual(noiseless, bestGuess)
