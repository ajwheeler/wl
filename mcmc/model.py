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

    r_psf = .25

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
        return "Disk{r=%s, I=%s, g1=%s, g2=%s}, Bulge{r=%s, I=%s, g1=%s, g2=%s}, Shear{g1=%s, g2=%s}, r_psf=%s, " \
            % (self.rd, self.fd, self.g1d, self.g2d, 
               self.rb, self.fb, self.g1b, self.g2b, 
                self.g1s, self.g2s, self.r_psf)

    def fromArray(self, array):
        if array.shape != (10,):
            raise RuntimeError("parameter array should be a numpy array with shape (10,)")
        self.rd = array[0]
        self.fd = array[1]
        self.g1d = array[2]
        self.g2d = array[3]

        self.rb = array[4]
        self.fb = array[5]
        self.g1b = array[6]
        self.g2b = array[7]

        self.g1s = array[8]
        self.g2s = array[9]

    def toArray(self):
        return np.array([self.rd, self.fd, self.g1d, self.g2d, self.rb, self.fb, self.g1b,
                         self.g2b, self.g1s, self.g2s])


def egg(params, scale=None, match_image_size=None, verbose=False, SNR=None):
    disk = galsim.Exponential(half_light_radius=params.rd, flux=params.fd)
    disk = disk.shear(g1=params.g1d, g2=params.g2d)
    disk = disk.withFlux(params.fd)

    bulge = galsim.DeVaucouleurs(half_light_radius=params.rb, flux=params.fb)
    bulge = bulge.shear(g1=params.g1b, g2=params.g2b)
    bulge = bulge.withFlux(params.fb)

    psf = galsim.Gaussian(sigma=params.r_psf)
    egg = disk + bulge

    #apply shear  
    egg = egg.shear(g1=params.g1s, g2=params.g2s)

    #convolve with point-spread function
    big_fft_params = galsim.GSParams(maximum_fft_size=10240)
    egg = galsim.Convolution(egg, psf, gsparams=big_fft_params)
    try:
        if match_image_size == None:
            image = egg.drawImage(scale=scale)
        else:
            image = egg.drawImage(scale=match_image_size.scale, bounds = match_image_size.bounds)

        if SNR != None:
            image.addNoiseSNR(galsim.GaussianNoise(rng=galsim.BaseDeviate(int(time.time()))),
                              SNR, preserve_flux=True)
    except RuntimeError:
        print("error with these parameters:")
        print(params)
        return -np.inf
    return image


    
