import utils
import numpy as np

def model(x0, y0, i, j, Id, Ib, rd, rb, ad, bd):
    modified_dist = np.sqrt( ((j-x0)/ad)**2 + ((i-y0)/bd)**2 )
    return Id*np.exp(-modified_dist/rd) + Ib*np.exp(-(modified_dist/rb)**(1/n))

def p_model_p_ad(x0, y0, i, j, Id, Ib, rd, rb, ad, bd):
    modified_dist = np.sqrt( ((j-x0)/ad)**2 + ((i-y0)/bd)**2 )
    if modified_dist == 0: return 0
    return Id*(j-x0)**2/rd * np.exp(-modified_dist/rd)/modified_dist * ad**(-3)

def p_model_p_bd(x0, y0, i, j, Id, Ib, rd, rb, ad, bd):
    modified_dist = np.sqrt( ((j-x0)/ad)**2 + ((i-y0)/bd)**2 )
    if modified_dist == 0: return 0
    return Id*(i-y0)**2/rd * np.exp(-modified_dist/rd)/modified_dist * bd**(-3)

def estimate_shear(image, x0=0, y0=0, Id=1, Ib=1, rd=1, rb=1, ad=1, bd=1):
    print("Starting guess: ad = %s, bd = %s" % (ad, bd))
    
    for _ in xrange(20):
        ad_numerator = 0
        bd_numerator = 0
        ad_denominator = 0
        bd_denominator = 0
        for i,j,val in utils.pixels(image):
            m = model(x0, y0, i, j, Id, Ib, rd, rb, ad, bd)
            pmpad = p_model_p_ad(x0, y0, i, j, Id, Ib, rd, rb, ad, bd)
            pmpbd = p_model_p_bd(x0, y0, i, j, Id, Ib, rd, rb, ad, bd)

            ad_numerator += (val-m)*pmpad
            bd_numerator += (val-m)*pmpbd

            ad_denominator += pmpad**2
            bd_denominator += pmpbd**2
        ad = ad_numerator/ad_denominator
        bd = bd_numerator/bd_denominator

        print("new guess: ad = %s, bd = %s" % (ad, bd))

    g = (ad-bd)/(ad+bd)
    print("g = " + str(g))

    
