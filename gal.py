import galsim
import numpy
import os

R0FLUX = 0.632121

def center(data):
    X = 0
    Y = 0
    totalflux = 0
    for x, col in enumerate(data):
        for y, val in enumerate(col):
            X += val*x
            Y += val*y
            totalflux += val
    return (X/totalflux,Y/totalflux)

def flux(data):
    flux = 0
    for col in data:
        for val in col:
            flux += val
    return flux

def mean_deviation(data, center):
    deviation_sum = 0
    total_flux  = 0
    X,Y = center
    for x, col in enumerate(data):
        for y, val in enumerate(col):
            radius = numpy.sqrt((x-X)**2 + (y-Y)**2)
            deviation_sum += radius*val
            total_flux += val

    return deviation_sum / total_flux

def std_deviation(data, center):
    deviation_sum = 0
    total_flux  = 0
    X,Y = center
    for x, col in enumerate(data):
        for y, val in enumerate(col):
            radiusSquared = (x-X)**2 + (y-Y)**2
            deviation_sum += radiusSquared*val
            total_flux += val
    return numpy.sqrt(deviation_sum / total_flux)

def radius_bounding_flux(data, center, targetFlux):
    fluxAtR = {}
    X,Y = center
    for x, col in enumerate(data):
        for y, val in enumerate(col):
            R = int(numpy.sqrt((x-X)**2 + (y-Y)**2))
            fluxAtR.setdefault(R, 0)
            fluxAtR[R] += val
    fluxSum = 0
    for R in sorted(fluxAtR):
        fluxSum += fluxAtR[R]
        if fluxSum >= targetFlux:
            return R
    return "targetFlux is too high, only saw %s total flux" % fluxSum
    

def regress(gal):
    print("Galaxy: %s" % gal)
    
    data = gal.drawImage().array
    
    #Initial parameter estimations
    A0 = center(data)
    r0 = radius_bounding_flux(data, A0, R0FLUX)
    I0 = data[A0]
    

def analyze(gal):
    print("Galaxy: %s" % gal)

    data = gal.drawImage().array
    origen = center(data)
    luminosity = flux(data)
    
    print("total flux: %s" % luminosity)
    print("center: %s" % (origen,))
    print("mean deviation from center: %s" % mean_deviation(data, origen))
    print("standard deviation from center: %s" % std_deviation(data, origen))
    print("1/2  light radius: %s" % radius_bounding_flux(data, origen, luminosity/2))
    print("90% light radius: " + str(radius_bounding_flux(data, origen, 9*luminosity/10)))
    print('')

if __name__ == '__main__':
    analyze(galgalsim.Exponential(scale_radius=3)

    random_seed = 8241573
    sky_level = 1.e6                # ADU / arcsec^2
    pixel_scale = 1.0               # arcsec / pixel  (size units in input catalog are pixels)
    gal_flux = 1.e6                 # arbitrary choice, makes nice (not too) noisy images
    gal_g1 = -0.009                 #
    gal_g2 = 0.011                  #
    xsize = 64                      # pixels
    ysize = 64                      # pixels

    cat = galsim.Catalog('/home/adam/GalSim/examples/input/galsim_default_input.asc')

    for k in range(cat.nobjects):
        rng = galsim.BaseDeviate(random_seed+k)

        # Take the Moffat beta from the first column (called 0) of the input catalog:
        # Note: cat.get(k,col) returns a string.  To get the value as a float, use either
        #       cat.getFloat(k,col) or float(cat.get(k,col))
        beta = cat.getFloat(k,0)
        # A Moffat's size may be either scale_radius, fwhm, or half_light_radius.
        # Here we use fwhm, taking from the catalog as well.
        fwhm = cat.getFloat(k,1)
        # A Moffat profile may be truncated if desired
        # The units for this are expected to be arcsec (or specifically -- whatever units
        # you are using for all the size values as defined by the pixel_scale).
        trunc = cat.getFloat(k,4)
        # Note: You may omit the flux, since the default is flux=1.
        psf = galsim.Moffat(beta=beta, fwhm=fwhm, trunc=trunc)

        # Take the (e1, e2) shape parameters from the catalog as well.
        psf = psf.shear(e1=cat.getFloat(k,2), e2=cat.getFloat(k,3))

        # Galaxy is a bulge + disk with parameters taken from the catalog:
        disk = galsim.Exponential(flux=0.6, half_light_radius=cat.getFloat(k,5))
        disk = disk.shear(e1=cat.getFloat(k,6), e2=cat.getFloat(k,7))

        bulge = galsim.DeVaucouleurs(flux=0.4, half_light_radius=cat.getFloat(k,8))
        bulge = bulge.shear(e1=cat.getFloat(k,9), e2=cat.getFloat(k,10))

        # The flux of an Add object is the sum of the component fluxes.
        # Note that in demo3.py, a similar addition was performed by the binary operator "+".
        gal = galsim.Add([disk, bulge])
        # This flux may be overridden by withFlux.  The relative fluxes of the components
        # remains the same, but the total flux is set to gal_flux.
        gal = gal.withFlux(gal_flux)
        gal = gal.shear(g1=gal_g1, g2=gal_g2)

        # The center of the object is normally placed at the center of the postage stamp image.
        # You can change that with shift:
        gal = gal.shift(dx=cat.getFloat(k,11), dy=cat.getFloat(k,12))

        final = galsim.Convolve([psf, gal])
        
        analyze(final)
        
