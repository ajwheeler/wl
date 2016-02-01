import numpy as np
import emcee
import matplotlib.pyplot as plt

chain = np.load("500.1000.3000.chain.npy")
f = emcee.autocorr.function(chain)

plt.plot(range(len(chain)), f[:,0])
plt.show()


print("25")
print(emcee.autocorr.integrated_time(chain, axis=0, window=25))
print(50)
print(emcee.autocorr.integrated_time(chain, axis=0, window=50))
print(100)
print(emcee.autocorr.integrated_time(chain, axis=0, window=100))
print(300)
print(emcee.autocorr.integrated_time(chain, axis=0, window=300))
print(1000)
print(emcee.autocorr.integrated_time(chain, axis=0, window=1000))
