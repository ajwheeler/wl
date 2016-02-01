import mcmc
import matplotlib.pyplot as plt
import numpy as np

theta = mcmc.np.array([-1.0])

domain = []
lnprobs = []
probs = []
while theta[0] < 1.0:
    domain.append(theta[0])
    lnprob = mcmc.lnprob(theta, mcmc.data, mcmc.trueParams.r_psf)
    lnprobs.append(lnprob)
    probs.append(np.exp(lnprob))
    theta[0] += 0.03


plt.plot(domain, lnprobs)
plt.show()

plt.plot(domain, probs)
plt.show()

plt.plot(domain, probs)
plt.plot(domain, lnprobs)
plt.show()
