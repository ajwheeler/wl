import galsim
import time
import numpy as np

def noisy_exp(scale_radius, flux, scale=1, sigma=1):
    """return a exponential gal image object with Gaussian noise"""
    perfect_gal = galsim.Exponential(scale_radius=scale_radius, flux=flux)
    image = perfect_gal.drawImage(scale=scale)
    image.addNoise(galsim.GaussianNoise(sigma=sigma, rng=galsim.BaseDeviate(int(time.time()))))
    return image

def analytic_I0_variance(data, center, r0, sigma):
    """the analytic variance of the I0 estimator for and exponential profile
    with r0 and center, and with Gaussian noise with standard deviation sigma.
    data is used purely to express the dimension of the image
    """
    numerator = 0
    root_denominator = 0
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            numerator += (sigma * np.exp(-dist/r0))**2
            root_denominator +=  np.exp(-2*dist/r0)
    return (numerator/(root_denominator**2))


def flux2I0(flux, scale_radius):
    """convert total flux to I0"""
    return flux / (2 * np.pi * scale_radius**2)

def I02flux(I0, scale_radius):
    """convert I0 to total flux"""
    return I0 * (2 * np.pi * scale_radius**2)
