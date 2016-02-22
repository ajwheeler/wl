import model
import numpy as np 
import matplotlib.pyplot as  plt

params = model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .01, g2s = .02)
data = model.egg(params, dual_band=False, scale=.2, nx=200, ny=200)
print(np.sum(data.array))
print(data.bounds)
plt.imshow(data.array, cmap=plt.get_cmap('gray'))
plt.show()
