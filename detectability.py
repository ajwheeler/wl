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

def brightest_pixel(data):
    p = data[0][0]
    for i,row in enumerate(data):
        for j,val in enumerate(row):
            if val > p:
                p = val
    return p

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
    minSNR = 1
    maxSNR = 20
    SNRStep = 1
    trialsPerStep = 10
    threshold = .6

    detectedPerSNR = {}
    brightestPixels = {}
    

    scale_radius = 5
    flux = 120

    snr = minSNR
    while snr <= maxSNR:
        print("snr = " + str(snr))
        detectedPerSNR[snr] = 0
        brightestPixels[snr] = 0

        for i in xrange(trialsPerStep):
            #print("iteration " + str(i))
            image = noisy_exp(scale_radius, flux, SNR=snr).array
            smoothed_image = convolve_with_model(image,scale_radius)

            centroid = find_galaxy(smoothed_image, threshold)
            if centroid:
                detectedPerSNR[snr] += 1
            
            bp = brightest_pixel(smoothed_image)
            print("brightest pixel: " + str(bp))
            brightestPixels[snr] += bp

            # print("view images?")
            # if get_y_n():
            #    plt.imshow(image, cmap=plt.get_cmap('gray'))
            #    plt.show()
            
            #    plt.imshow(smoothed_image, cmap=plt.get_cmap('gray'))
            #    plt.show()
        

        brightestPixels[snr] /= float(trialsPerStep)
        snr += SNRStep

    SNRs = sorted(detectedPerSNR.keys())
    detected = [detectedPerSNR[k]/float(trialsPerStep) for k in SNRs]

    plt.plot(SNRs, detected)
    plt.xlabel("SNR")
    plt.ylabel("probability of detection")
    plt.title("Liklyhood of detection (threshold: I0 = %s)" % threshold)
    plt.show()


    plt.plot(SNRs, [brightestPixels[k] for k in SNRs])
    plt.xlabel("SNR")
    plt.ylabel("avg brightest pixel")
    plt.show()
       
