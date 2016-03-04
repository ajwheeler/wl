import galsim
import numpy as np

class EggParams():
    rd = 3
    fd = .7
    g1d = 0
    g2d = 0

    rb = 1
    fb = .3
    g1b = 0 
    g2b = 0 

    g1s = 0
    g2s = 0

    mu = 1

    labels = ['rd','fd','g1d','g2d','rb','fb','g1b','g2b','g1s','g2s', 'mu']
    nparams = len(labels)

    def __init__(self, **params):
        for k in params:
            if hasattr(self, k):
                setattr(self, k, params[k])
            else:
                raise AttributeError("EggParams has no attribute " + k)

            setattr(self, k, params[k])

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, k, value):
        if hasattr(self, k):
            setattr(self, k, value)
        else:
            raise AttributeError("EggParams has no attribute " + k)


    def __repr__(self):
        return "Disk{r=%s, I=%s, g1=%s, g2=%s}, Bulge{r=%s, I=%s, g1=%s, g2=%s}, Lensing{g1=%s, g2=%s, mu=%s}"\
            % (self.rd, self.fd, self.g1d, self.g2d, 
               self.rb, self.fb, self.g1b, self.g2b, 
               self.g1s, self.g2s, self.mu)

    def fromArray(self, array, mask=[True]*nparams):
        if array.shape != (mask.count(True),):
            raise RuntimeError("parameter array should be a numpy array with shape (%s,)" 
                               % mask.count(True))
             
        j = 0
        for i in xrange(self.nparams):
            if mask[i]:
                self[self.labels[i]] = array[j]
                j += 1
        

    def toArray(self, mask=[True]*nparams):
        vals = []
        for i in xrange(self.nparams):
            if mask[i]:
                vals.append(self[self.labels[i]])
        return np.array(vals)


def egg(params, scale=None, match_image_size=None, dual_band=True, nx=None, ny=None):
    r_psf = .25

    disk = galsim.Exponential(half_light_radius=params.rd, flux=params.fd)
    disk = disk.shear(g1=params.g1d, g2=params.g2d)
    disk = disk.withFlux(params.fd)

    bulge = galsim.DeVaucouleurs(half_light_radius=params.rb, flux=params.fb)
    bulge = bulge.shear(g1=params.g1b, g2=params.g2b)
    bulge = bulge.withFlux(params.fb)

    if dual_band:
        rfr = 5.0**.25
        green_egg = (disk*rfr + bulge/rfr)
        red_egg = (disk/rfr + bulge*rfr)

        green_egg  = green_egg.withFlux(params.fd + params.fb)
        red_egg  = red_egg.withFlux(params.fd + params.fb)
    else:
        egg = disk + bulge

    images = []
    for egg in [green_egg, red_egg] if dual_band else [egg]:
        #apply shear  
        egg = egg.shear(g1=params.g1s, g2=params.g2s)
        egg = egg.magnify(params.mu)

        #convolve with point-spread function
        big_fft_params = galsim.GSParams(maximum_fft_size=10240)
        psf = galsim.Gaussian(sigma=r_psf)
        egg = galsim.Convolution(egg, psf, gsparams=big_fft_params)
        if match_image_size == None:
            image = egg.drawImage(scale=scale, nx=nx, ny=ny)
        else:
            image = egg.drawImage(scale=match_image_size.scale, 
                                  bounds = match_image_size.bounds)
        images.append(image)

    return (images[0], images[1]) if dual_band else images[0]


    
