import priors
import numpy as np

print priors.kormendy_prior(1.0,.3)

Re = 1.0
F = .3
mu0 = 20.3719193183 - 2.5 * np.log10(2 * np.pi * Re**2 / F)
print mu0
print

# import sys
# sys.exit()
# print priors.kormendy_prior(1,.1)
# print
# print priors.kormendy_prior(1,1)
#
# print
# print
# print priors.kormendy_prior(1.1,.3)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Rs = []
Fs = []
probs = []
for R in range(1,20):
    R /= 10.0
    for F in range(1,10,1):
        F /= 10.0
        Rs.append(R)
        Fs.append(F)
        probs.append(priors.kormendy_prior(R,F))

ax.scatter(Rs, Fs, probs)
ax.set_xlabel("R")
ax.set_ylabel("F")
ax.set_zlabel("prob")
plt.show()
