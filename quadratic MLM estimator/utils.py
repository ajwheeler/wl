import galsim
import time
import numpy as np
import matplotlib.pyplot as  plt

def noisy_exp(scale_radius, flux, scale=1, sigma=1, SNR=None):
    """return a exponential gal image object with Gaussian noise"""
    perfect_gal = galsim.Exponential(scale_radius=scale_radius, flux=flux)
    image = perfect_gal.drawImage(scale=scale)
    if SNR == None:
        image.addNoise(galsim.GaussianNoise(sigma=sigma, rng=galsim.BaseDeviate(int(time.time()))))
    else:
        image.addNoiseSNR(galsim.GaussianNoise(rng=galsim.BaseDeviate(int(time.time()))), SNR, preserve_flux=False)

    return image

def noisy_egg(rd, Id, rb, Ib, disk_g=0, scale=None, sigma=0):
    """draw a noisy image of a disk + bulge galaxy
    the disk has scale radius rd and intensity at center Id
    the bulge has scale radius rb and intensity at center Ib"""

    disk_flux = 2*Id*np.pi*rd**2
    bulge_flux = 40320*Ib*np.pi*(rb**2)

    print("disk_flux = " + str(disk_flux))
    print("bulge_flux = " + str(bulge_flux))

    disk = galsim.Exponential(half_light_radius=rd, flux=disk_flux)
    disk = disk.shear(g1=disk_g)
    bulge = galsim.DeVaucouleurs(half_light_radius=rb, flux=bulge_flux)

    #todo base on half light radius
    psf = galsim.Gaussian(sigma=min([rd, rb])/5.0)

    egg = galsim.Convolution([(disk+bulge), psf])
    #image = egg.drawImage(scale=scale)
    image = disk.drawImage(scale=scale)
    image.addNoise(galsim.GaussianNoise(sigma=sigma, rng=galsim.BaseDeviate(int(time.time()))))
    
    return (image.array, image.center())
    

def analytic_I0_variance(data, center, r0):
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
    """convert total flux to I0 for exponential profile"""
    return flux / (2 * np.pi * scale_radius**2)

def I02flux(I0, scale_radius):
    """convert I0 to total flux for exponential profile"""
    return I0 * (2 * np.pi * scale_radius**2)


def view_image(image):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()

def pixels(image):
    for i, row in enumerate(image):
        for j, val in enumerate(row):
            yield i,j,val
