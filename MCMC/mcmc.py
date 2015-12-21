import model
import emceee
import numpy as np

def lnprob(theta, data, psf):
    params = model.EggParams(psf=psf)
    params.fromArray(theta)
    gal = model.egg(params, match_image_size=data)
    diff = gal.array - data.array
    return np.sum(diff**2)

nwalkers = 100

