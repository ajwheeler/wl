import utils
import numpy as np

def modified_distance(x0, y0, i, j, a, b):
    return np.sqrt( ((x0-j)/a)**2 + ((y0-i)/b)**2)

def model(x0, y0, i, j, I, a, b):
    return I*np.exp(-modified_distance(x0, y0, i, j, a, b))

def p_model_p_a(x0, y0, i, j, I, a, b):
    dist = modified_distance(x0, y0, i, j, a, b)
    if dist==0:
        return 0
    return I*np.exp(-dist)*(j-x0)**2/(a**3 * dist)

def p_model_p_b(x0, y0, i, j, I, a, b):
    dist = modified_distance(x0, y0, i, j, a, b)
    if dist==0:
        return 0
    return I*np.exp(-dist)*(i-y0)**2/(b**3 * dist)

def estimate_shear(image, x0=0, y0=0, I=1, a=1, b=1):
    print("Starting guess: a = %s, b = %s" % (a, b))

    for _ in xrange(10):
        a_num = 0
        a_denom = 0
        b_num = 0
        b_denom = 0

        for i,j,val in utils.pixels(image):
            m = model(x0, y0, i, j, I, a, b)
            pmpa = p_model_p_a(x0, y0, i, j, I, a, b)
            pmpb = p_model_p_b(x0, y0, i, j, I, a, b)

            a_num += (val-m)*pmpa
            b_num += (val-m)*pmpb

            a_denom += pmpa**2
            b_denom += pmpb**2
        a += a_num/a_denom
        b += b_num/b_denom

        print("new guess: a = %s, b = %s" % (a,b))
        
    g = (a-b)/(a+b)
    print("g = " + str(g))
    return g

