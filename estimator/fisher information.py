import numerical_elipticity_estimator as nee
import numpy as np
import utils

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ORTHOGONALIZED_ELIPTICITY_SCORE = False
SNR = 50
np.set_printoptions(precision=4, linewidth=150, suppress=True)


if ORTHOGONALIZED_ELIPTICITY_SCORE:
    print("Using orthogonalized elipticity score")
else:
    print("Not using orthogonalized scores")


def plot_elipse(params, cov, labels, p1, p2, verbose=False):
    i = labels.index(p1)
    j = labels.index(p2)

    theta = .5 * np.arctan(2*cov[i,j]/(cov[i,i]-cov[j,j]))
    theta *= 180.0/np.pi

    alpha = 1.52
    a = np.sqrt((cov[i,i]+cov[j,j])/2 + np.sqrt((cov[i,i]-cov[j,j])**2/4 + cov[i,j]**2))*alpha
    b = np.sqrt((cov[i,i]+cov[j,j])/2 - np.sqrt((cov[i,i]-cov[j,j])**2/4 + cov[i,j]**2))*alpha

    center = (getattr(params,p1), getattr(params,p2))
    
    lim = max(a,b) * .6
    xlims = (center[0]-lim, center[0]+lim)
    ylims = (center[1]-lim, center[1]+lim)
    
    if verbose:
        print("center = %s" % (center,))
        print("a = %s" % a)
        print("b = %s" % b)
        print("theta = %s" % theta)
        print("xlims = %s" % (xlims,))
        print("ylims = %s" % (ylims,))
    
    ellipse = Ellipse(xy=center,width=a, height=b, angle=theta)
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='auto')
    ax.add_artist(ellipse)

    ellipse.set_clip_box(ax.bbox)
  
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

    ax.set_xlabel(params.names[p1])
    ax.set_ylabel(params.names[p2])
    plt.show()


params = nee.EggParams()
egg = nee.noisy_egg(params)

labels = ['rd','fd','g1d','g2d','rb','fb','g1b','g2b','g1s','g2s']

derivatives = {}
for p in labels:
    if ORTHOGONALIZED_ELIPTICITY_SCORE and p in ['g1d', 'g2d', 'g1b', 'g2b']:
        dmdg = nee.dmd(egg, params, 'g'+p[1]+'s')
        dmdp = nee.dmd(egg, params, p)

        dmdp -= np.sum(dmdg * dmdp)/np.sum(dmdg**2) * dmdg
        
        derivatives[p] = dmdp
        
    else:
        derivatives[p] = nee.dmd(egg, params, p)

fisher = np.zeros((len(labels), len(labels)))
for i,l1 in enumerate(labels):
    for j, l2 in enumerate(labels):
        fisher[i,j] = np.sum(derivatives[l1]*derivatives[l2])
fisher = np.matrix(fisher)*(SNR**2)
covariance = np.linalg.inv(fisher)

print(labels)
print("model fisher information")
print(fisher)
#print("model covariance")
#print(covariance)
#plot_elipse(params, covariance, labels, 'g1s', 'g1d')


def orientation_prior(g):
    p = 1/(4-np.pi) * np.sqrt(g)/(1+g)
    #print("p(%s) = %s" % (g,p))
    return p
    #return 1 if g > .3 else 0

accepted = []
allPoints = []
for _ in xrange(50000):
    point = np.random.multivariate_normal([params[l] for l in labels], covariance)

    g1 = np.tanh(point[labels.index('g1d')])
    g2 = np.tanh(point[labels.index('g2d')])
    g = np.sqrt(g1**2 + g2**2)
    #print("g1 = %s, g2 = %s, g = %s" % (g1,g2,g))
    p = orientation_prior(g)
    r = np.random.random()
    #print("%s < %s ?" %(r, p))
    if r < p:
        accepted.append(point)
    allPoints.append(point)

points_cov = np.cov(np.transpose(allPoints))
points_fisher = np.linalg.inv(points_cov)
print("points fisher")
print(points_fisher)

print(str(len(accepted)) + " points accepted")
covariance_with_prior = np.cov(np.transpose(accepted))
fisher_with_prior = np.linalg.inv(covariance_with_prior)
print("model + prior fisher")
print(fisher_with_prior)

print("prior fisher")
print(fisher_with_prior - fisher)

print("true prior fisher")
print(fisher_with_prior - points_fisher)

#plot_elipse(params, covariance_with_prior, labels, 'g1s', 'g1d')
print()


#conditionals  = [] 
#marginalizeds = []
#for i in xrange(10):
#    conditionals.append(1.0/np.sqrt(fisher[i,i]))
#    marginalizeds.append(np.sqrt(covariance[i,i]))
#
#for t in zip(labels, conditionals, marginalizeds):
#    print(t)



#correlation matrix for g1d, g1s, fd
#correlation = np.zeros((4,4))
#
#interesting_indices = [labels.index(l) for l in ['g1d', 'g1b', 'g1s', 'fd']]
#for i,iP in enumerate(interesting_indices):
#   for j,jP in enumerate(interesting_indices):
#       correlation[i,j] = covariance[iP,jP]/np.sqrt(covariance[iP,iP] * covariance[jP,jP])
#
#print(correlation)
