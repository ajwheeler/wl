import utils
import numpy as np

def modified_distance(x0, y0, i, j, a, xi):
    return np.sqrt((x0-j)**2 + ((y0 - i)/xi)**2)/a

def model(x0, y0, i, j, Id, Ib, ad, ab, xid, xib):
    d_dist = modified_distance(x0, y0, i, j, ad, xid)
    b_dist = modified_distance(x0, y0, i, j, ab, xib)
    return Id*np.exp(-d_dist) + Ib*np.exp(-b_dist**(1/4))

def p_model_p_ad(x0, y0, i, j, Id, Ib, ad, ab, xid, xib):
    dist = modified_distance(x0, y0, i, j, ad, xid)
    return Id*np.exp(-dist)*dist/ad

def p_model_p_xid(x0, y0, i, j, Id, Ib, ad, ab, xid, xib):
    dist = modified_distance(x0, y0, i, j, ad, xid)
    if i == y0:
        return 0
    return Id*np.exp(-dist)*(i - y0)**2/(dist * ad**2 * xid**3)

def estimate_shear(image, x0=0, y0=0, Id=1, Ib=1, ad=1, ab=1, xid=1, xib=1):
    print("Starting guess: ad = %s, xid = %s" % (ad, xid))
    
    for _ in xrange(100):
        ad_numerator = 0
        xid_numerator = 0
        ad_denominator = 0
        xid_denominator = 0
        for i,j,val in utils.pixels(image):
            m = model(x0, y0, i, j, Id, Ib, ad, ab, xid, xib)
            pmpad = p_model_p_ad(x0, y0, i, j, Id, Ib, ad, ab, xid, xib)
            pmpxid = p_model_p_xid(x0, y0, i, j, Id, Ib, ad, ab, xid, xib)

            #if j-x0==5 and i-y0==2:
            #    print("m = %s, pmpad = %s, pmpxid = %s" %(m,pmpad, pmpxid))

            ad_numerator += (val-m)*pmpad
            xid_numerator += (val-m)*pmpxid

            ad_denominator += pmpad**2
            xid_denominator += pmpxid**2
        ad += ad_numerator/ad_denominator
        xid += xid_numerator/xid_denominator

        print("new guess: ad = %s, xid = %s" % (ad, xid))

    g = (1-xid)/(1+xid)
    print("g = " + str(g))
    return g

    
