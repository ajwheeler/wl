import numpy as np
import matplotlib.pyplot as plt

chain = np.load("50.100.250.1-25.chain.npy")

paramPos = 2 #g1d

for paramPos in xrange(10):
    trace = chain[:,paramPos]

    print(chain.shape)
    print(trace.shape)

    plt.plot(range(len(chain)), trace)
    plt.show()
