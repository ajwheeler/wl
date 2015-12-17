import galsim
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import time 
import utils

R0FLUX = 0.632121

def distance(i,j,point):
    i0, j0 = point
    return np.sqrt((i-i0)**2 + (j-j0)**2)

def center(data):
    X = 0
    Y = 0
    totalflux = 0
    for y, row in enumerate(data):
        for x, val in enumerate(row):
            X += val*x
            Y += val*y
            totalflux += val
    return (Y/totalflux,X/totalflux)

def radius_bounding_flux(data, center, targetFlux):
    fluxAtR = {}   
    for y, row in enumerate(data):
        for x, val in enumerate(row):
            R = int(distance(x,y,center))
            fluxAtR.setdefault(R, 0)
            fluxAtR[R] += val
    fluxSum = 0
    for R in itertools.count():
        if R not in fluxAtR:
            print(fluxAtR)
            raise RuntimeError("targetFlux is too high, only saw %s total flux" % fluxSum)
        fluxSum += fluxAtR[R]
        if fluxSum >= targetFlux:
            return R if R != 0 else 1

def model(i, j, A, r0, I0):
    return I0 * np.exp(- distance(i,j,A)/r0)

def d_model_d_A(i, j, A, r0, I0):
    delt = distance(i,j,A)
    m = I0*np.exp(-delt/r0)/(r0*delt)
    Ai, Aj = A
    return (m*(i-Ai), m*(j-Aj))

def d_model_d_I0(i, j, A, r0, I0):
    return np.exp(- distance(i,j,A)/r0)

def d_model_d_r0(i, j, A, r0, I0):
    delt = distance(i,j,A)
    return I0*np.exp(-delt/r0)*(delt/(r0**2))

def log_likelihood(data, A, r0, I0):
    s = 0
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            s += (data[i,j] - model(i,j,A,r0,I0))**2
    return -s
            
def get_y_n():
    c = raw_input("[y]/n: ")
    if c == 'y' or c == '':
        return True
    if c == 'n':
        return False
    else:
        return get_y_n()

def regress_step(data, A, r0, I0, output=True):
    #factors in improved estimator formulas
    Ai_numerator = 0
    Ai_denominator = 0
    Aj_numerator = 0
    Aj_denominator = 0
    r0_numerator = 0
    r0_denominator = 0
    I0_numerator = 0
    I0_denominator = 0
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            #Di - Mi, differemce between data and model
            DMDiff = val - model(i,j,A,r0,I0)

            dmdAi, dmdAj = d_model_d_A(i,j,A,r0,I0)
            Ai_numerator += DMDiff*dmdAi
            Ai_denominator += dmdAi**2
            Aj_numerator += DMDiff*dmdAj
            Aj_denominator += dmdAj**2

            dmdr0 = d_model_d_r0(i,j,A,r0,I0)
            r0_numerator += DMDiff*dmdr0
            r0_denominator += dmdr0**2

            dmdI0 = d_model_d_I0(i,j,A,r0,I0)
            I0_numerator += DMDiff*dmdI0
            I0_denominator += dmdI0**2

    A = (A[0]+Ai_numerator/Ai_denominator, A[1]+Aj_numerator/Aj_denominator) 
    r0 += r0_numerator / r0_denominator
    I0 += I0_numerator / I0_denominator
    
    if output:
        print("A = " + str(A))
        print("r0 = " + str(r0))
        print("I0 = " + str(I0))
        print("log likelihood: " + str(log_likelihood(data, A, r0, I0)))

    return (A, r0, I0)

def regress(data, interactive=False, iterations=5, output=True):
    #Initial parameter estimations
    f = data.sum()
    A = center(data)
    A = (A[0], A[1])
    r0 = radius_bounding_flux(data, A, f*R0FLUX)
    I0 = np.absolute(f / (2.0 * np.pi * r0**2))

    if output:
        print("initial values")
        print("A = " + str(A))
        print("r0 = " + str(r0))
        print("I0 = " + str(I0))
        print("log likelihood: " + str(log_likelihood(data, A, r0, I0)))
    if interactive:
        while True:
            print("Do another iteration?")
            if not get_y_n():
                break
            A, r0, I0 = regress_step(data, A, r0, I0, output=output)
    else:
        for iteration in xrange(iterations):
            A, r0, I0 = regress_step(data, A, r0, I0, output=output)
    return (A, r0, I0)

def noisy_exp(scale_radius, flux, scale=1):
    perfect_gal = galsim.Exponential(scale_radius=scale_radius, flux=flux)
    image = perfect_gal.drawImage(scale=scale)
    image.addNoise(galsim.GaussianNoise(sigma=.05, rng=galsim.BaseDeviate(int(time.time()))))
    return image

if __name__ == '__main__':
    
    psf = galsim.Gaussian(sigma=2)

    perfect_gal = galsim.Exponential(scale_radius=5, flux=100)
    perfect_gal = galsim.Convolution([perfect_gal, psf])
    image = perfect_gal.drawImage(scale=1)
    utils.view_image(image.array)
    image.addNoise(galsim.GaussianNoise(sigma=.1, rng=galsim.BaseDeviate(int(time.time()))))
    utils.view_image(image.array)
    data = image.array
    A,r0,I0 = regress(data, iterations=30)
    print("estimated I0: " + str(I0))
        
    print('plotting')
    image = plt.imshow(data, cmap=plt.get_cmap('gray'))
    print('adding centroid at ' + str(A))
    plt.scatter(A[0],A[1])

    plt.show()

    # cat = galsim.Catalog('/home/adam/GalSim/examples/input/galsim_default_input.asc')
    # for k in range(cat.nobjects):
    #     beta = cat.getFloat(k,0)
    #     fwhm = cat.getFloat(k,1)
    #     trunc = cat.getFloat(k,4)
    #     psf = galsim.Moffat(beta=beta, fwhm=fwhm, trunc=trunc)
    #     psf = psf.shear(e1=cat.getFloat(k,2), e2=cat.getFloat(k,3))
    #     disk = galsim.Exponential(flux=0.6, half_light_radius=cat.getFloat(k,5))
    #     disk = disk.shear(e1=cat.getFloat(k,6), e2=cat.getFloat(k,7))
    #     bulge = galsim.DeVaucouleurs(flux=0.4, half_light_radius=cat.getFloat(k,8))
    #     bulge = bulge.shear(e1=cat.getFloat(k,9), e2=cat.getFloat(k,10))
    #     gal = galsim.Add([disk, bulge])
    #     gal_flux = 1.e6
    #     gal_g1 = -0.009
    #     gal_g2 = 0.011
    #     gal = gal.withFlux(gal_flux)
    #     gal = gal.shear(g1=gal_g1, g2=gal_g2)
    #     gal = gal.shift(dx=cat.getFloat(k,11), dy=cat.getFloat(k,12))
    #     final = galsim.Convolve([psf, gal])        
    #     regress(final)
    

