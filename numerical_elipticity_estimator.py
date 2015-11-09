import utils
import galsim
import numpy as np
import time

class EggParams():
    rd = 3
    fd = .7
    g1d = 0
    g2d = 0

    rb = 1
    fb = .3
    g1b = 0
    g2b = 0 

    g1g = 0
    g2g = 0

    r_psf = .25

    def __init__(self, **params):
        for k in params:
            setattr(self, k, params[k])

    def __repr__(self):
        return "Disk{r=%s, I=%s, g1=%s, g2=%s}, Bulge{r=%s, I=%s, g1=%s, g2=%s}, r_psf=%s, g1g=%s, g2g=%s" \
            % (self.rd, self.fd, self.g1d, self.g2d, 
               self.rb, self.fb, self.g1b, self.g2b, 
               self.r_psf, self.g1g, self.g2g)
        
    
    
def noisy_egg(params, scale=None, match_image=None, verbose=False, sigma=0):
    #flux_d = 2*params.Id*np.pi*params.rd**2
    disk = galsim.Exponential(half_light_radius=params.rd, flux=params.fd)
    disk = disk.shear(g1=params.g1d, g2=params.g2d)

    #flux_b = 40320*params.Ib*np.pi*(params.rb**2)
    #TODO: half light radius from scale radius of de Vaucouleur profile
    bulge = galsim.DeVaucouleurs(half_light_radius=params.rb, flux=params.fb)
    bulge = bulge.shear(g1=params.g1b, g2=params.g2d)

    psf = galsim.Gaussian(sigma=params.r_psf)
    egg = disk + bulge
    egg = egg.shear(g1=params.g1g, g2=params.g2g)
    egg = galsim.Convolution(egg, psf)

    if match_image == None:
        image = egg.drawImage(scale=scale)
    else:
        image = egg.drawImage(scale=match_image.scale, bounds = match_image.bounds)

    image.addNoise(galsim.GaussianNoise(
        sigma=sigma, rng=galsim.BaseDeviate(int(time.time()))))

    return image

def dmd(egg, params, p):
    epsilon_factor = 1000.0
    delta_p = max([getattr(params,p)/epsilon_factor, .0000001])
    newparams = params
   
    setattr(newparams, p,  getattr(params,p) - delta_p)
    left_dmdp = noisy_egg(newparams, match_image=egg).array
    
    setattr(newparams, p, getattr(params,p) + delta_p)
    right_dmdp = noisy_egg(newparams, match_image=egg).array
    
    return (right_dmdp - left_dmdp)/(2*delta_p)*10

def estimate(egg, params, to_estimate=['rd','fd','g1d','g2d', 'rb','fb','g1b','g2b']):
    print("Starting guess: " + str(params))

    for i in xrange(501):
        for p in to_estimate:
            dmdp = dmd(egg, params,p)

            model = noisy_egg(params, match_image=egg).array
            
            p_correction = np.sum(dmdp*(egg.array - model))/np.sum(dmdp**2)
            #params[p] += p_correction
            setattr(params, p, getattr(params,p) + p_correction)
        if i % 20 == 0:
            print("iteration %s: %s" % (i,params))


if __name__ == '__main__':
    params = EggParams(g1d=.1, g2d=.1, g1g=0.01)
    egg = noisy_egg(params)
    utils.view_image(egg.array)
    
    params = EggParams()
    estimate(egg, params, to_estimate=['g1d', 'g2d', 'g1g', 'g2g'])


    params = EggParams()
    egg = noisy_egg(params)

    dmdgamma = dmd(egg, params, 'g1g')
    utils.view_image(dmdgamma)

    dmdIA = dmd(egg, params, 'g1d')
    utils.view_image(dmdIA)

    
    dmdgammaP = dmdgamma - (np.sum(dmdgamma * dmdIA)/np.sum(dmdIA**2))*dmdIA
    utils.view_image(dmdgammaP)
