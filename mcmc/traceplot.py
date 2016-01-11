import numpy as np
import matplotlib.pyplot as plt

chain = np.load("500.1000.3000.chain.npy")

paramPos = 2 #g1d

for paramPos in xrange(10):
    trace = chain[:,paramPos]

    print(chain.shape)
    print(trace.shape)

    plt.plot(range(len(chain)), trace)
    plt.show()
