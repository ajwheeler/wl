import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from utils import noisy_exp
import centroid_regression
import I0_estimator


#TODO: don't assume there is one galaxy
def find_galaxy(data, threshold):
    brightPixels = []
    for i,row in enumerate(data):
        for j,val in enumerate(row):
            if val > threshold:
                brightPixels.append((i,j))
    
    if brightPixels == []:
        return None

    X = 0
    Y = 0
    for x,y in brightPixels:
        X += x
        Y += y
    X /= len(brightPixels)
    Y /= len(brightPixels)
    
    #print("possible centroid: " + str((X,Y)))
    #print("intensity at center: " + str(data[X,Y]))

    return(X,Y)

def convolve_with_model(image, r0):
    """return the image convoled with an exponential profile with scale radius r0"""
    rows = len(image)
    cols = len(image[1])

    modelImage = np.empty((rows,cols))
    center = (rows/2.0, cols/2.0)
    for i in xrange(rows):
        for j in xrange(cols):
            distSquared = (i-center[0])**2 + (j - center[1])**2
            modelImage[i,j] = np.exp(-distSquared/r0)/(np.pi * r0)

    totalF = 0
    for i in xrange(rows):
        for j in xrange(cols):
            totalF += modelImage[i,j]

    return(ndimage.convolve(image, modelImage))

def get_y_n():
    c = raw_input("[y]/n: ")
    if c == 'y' or c == '':
        return True
    if c == 'n':
        return False
    else:
        return get_y_n()


if __name__ == '__main__':
    minFlux = 110
    maxFlux = 125
    fluxStep = 1
    trialsPerStep = 100
    threshold = .5

    detectedPerFlux = {}

    scale_radius = 5
    sigma = .05

    flux = minFlux
    while flux <= maxFlux:
        print("flux = " + str(flux))
        detectedPerFlux[flux] = 0

        for i in xrange(trialsPerStep):
            #print("iteration " + str(i))
            image = noisy_exp(scale_radius, flux, sigma=sigma).array
            smoothed_image = convolve_with_model(image,scale_radius)

            centroid = find_galaxy(smoothed_image, threshold)
            if centroid:
                detectedPerFlux[flux] += 1
            
            #print("view images?")
            if False:#get_y_n():
                plt.imshow(image, cmap=plt.get_cmap('gray'))
                plt.show()

                plt.imshow(smoothed_image, cmap=plt.get_cmap('gray'))
                plt.show()
                        
        flux += fluxStep

    image = noisy_exp(scale_radius, flux, sigma=sigma).array
    variance = I0_estimator.analytic_variance(image, (42.5, 42.5), scale_radius, sigma)

    fluxes = sorted(detectedPerFlux.keys())
    detected = [detectedPerFlux[k]/float(trialsPerStep) for k in fluxes]
    SNRs = [f/(np.sqrt(variance)*2*np.pi*(scale_radius**2)) for f in fluxes]
    plt.plot(SNRs, detected)
    plt.xlabel("SNR")
    plt.ylabel("probability of detection")
    plt.title("Liklyhood of detection (threshold: I0 = %s)" % threshold)
    plt.show()
