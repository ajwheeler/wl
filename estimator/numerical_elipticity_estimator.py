import utils
import galsim
import numpy as np
import time

class EggParams():
    # NB: gX is arctanh(gamma_X)
    rd = 3
    fd = .7
    gd = 0
    bd = 0

    rb = 1
    fb = .3
    gb = 0 
    bb = 0 

    gs = 0
    bs = 0

    r_psf = .25

    names = {
        'rd': 'disk radius (arcseconds)',
        'fd': 'toal disk flux',
        'g1d': 'disk IA (arctanh(gamma)) (1)',
        'g2d': 'disk IA (arctanh(gamma)) (2)',
        'rb': 'bulge radius (arcseconds)',
        'fb': 'total bulge flux',
        'g1b': 'bulge IA (arctanh(gamma)) (1)',
        'g2b': 'bulge IA (arctanh(gamma)) (2)',
        'g1s': 'shear (arctanh(gamma)) (1)',
        'g2s': 'shear (arctanh(gamma)) (2)'
    }

    def __init__(self, **params):
        for k in params:
            if hasattr(self, k):
                setattr(self, k, params[k])
            else:
                raise AttributeError("EggParams has no attribute " + k)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __repr__(self):
        return "Disk{r=%s, I=%s, g=%s, beta=%s}, Bulge{r=%s, I=%s, g=%s, beta=%s}, Shear{g=%s, beta=%s}, r_psf=%s, " \
            % (self.rd, self.fd, self.gd, self.bd, 
               self.rb, self.fb, self.gb, self.bb, 
                self.gs, self.bs, self.r_psf)
        
    
    
def noisy_egg(params, scale=None, match_image=None, verbose=False, SNR=None):
    disk = galsim.Exponential(half_light_radius=params.rd, flux=params.fd)
    disk = disk.shear(g=abs(np.tanh(params.gd)), beta=params.bd*galsim.radians)
    disk = disk.withFlux(params.fd)

    bulge = galsim.DeVaucouleurs(half_light_radius=params.rb, flux=params.fb)
    bulge = bulge.shear(g=abs(np.tanh(params.gb)), beta=params.bb*galsim.radians)
    bulge = bulge.withFlux(params.fb)

    psf = galsim.Gaussian(sigma=params.r_psf)
    egg = disk + bulge

    #apply shear  
    egg = egg.shear(g=abs(np.tanh(params.gs)), beta=params.bs*galsim.radians)

    #convolve with point-spread function
    egg = galsim.Convolution(egg, psf)

    if match_image == None:
        image = egg.drawImage(scale=scale)
    else:
        image = egg.drawImage(scale=match_image.scale, bounds = match_image.bounds)

    if SNR != None:
        image.addNoiseSNR(galsim.GaussianNoise(rng=galsim.BaseDeviate(int(time.time()))),
                          SNR, preserve_flux=True)
    return image

def dmd(egg, params, p, epsilon_factor = 1000.0, min_delta = .0000001):
    delta_p = max([getattr(params,p)/epsilon_factor, min_delta])
    newparams = params
   
    setattr(newparams, p,  getattr(params,p) - delta_p)
    left_dmdp = noisy_egg(newparams, match_image=egg).array
    
    setattr(newparams, p, getattr(params,p) + delta_p)
    right_dmdp = noisy_egg(newparams, match_image=egg).array
    
    return (right_dmdp - left_dmdp)/(2*delta_p)

#TODO orthogonal sheer adjustment for g2 s
def estimate(egg,params,to_estimate=['rd','fd','gd','bd','rb','fb','gb','bb','gs','bs'],
             use_orthogonal_sheer=False, step_scale=10.0):
    print("Starting guess: " + str(params))

    for i in xrange(501):
        for p in to_estimate:
            dmdp = dmd(egg, params,p)
            model = noisy_egg(params, match_image=egg).array
            p_correction = np.sum(dmdp*(egg.array - model))/np.sum(dmdp**2)/step_scale
            setattr(params, p, getattr(params,p) + p_correction)
        if i % 20 == 0:
            print("iteration %s: %s" % (i,params))


if __name__ == '__main__':
    params = EggParams(gd=.1, bd=1, gs=0.01)
    egg = noisy_egg(params)
    utils.view_image(egg.array)
    
    params = EggParams()
    estimate(egg, params, to_estimate=['gd', 'bd', 'gs', 'bs'], step_scale=100.0)
