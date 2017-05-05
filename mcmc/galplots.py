# draw some plots for the Denman poster
# coding: utf-8

import model
import galsim
import matplotlib.pyplot as  plt
params = model.EggParams(g1d = .2, g2d = .3, g2b = .4)
params = model.EggParams(g1d = .2, g2d = .3, g2b = .4, g1s = .02, g2s = .04, mu = 1.04)
i = model.egg(params, scale=.2, nx=100,ny=100,dual_band=False)
plt.imshow(i.array, cmap=plt.get_cmap('gray'))
plt.show()
plt.imshow(i.array, cmap=plt.get_cmap('gray'))
plt.savefig("unlensed_nonoise.png")
lensed = model.egg(params, scale=.2, nx=100,ny=100,dual_band=False)
params = model.EggParams(g1d = .2, g2d = .3, g2b = .4)
unlensed = model.egg(params, scale=.2, nx=50,ny=50,dual_band=False)
params
params.mu = 1.04
params.g1s = 1.02
params.g2s = 1.04
lensed = model.egg(params, scale=.2, nx=50,ny=50,dual_band=False)
params.g1s = .02
params.g2s = .04
params
lensed = model.egg(params, scale=.2, nx=50,ny=50,dual_band=False)
lensed
unlensed
diff = lensed.array - unlensed.array
diff
plt.imshow(lensed.array, cmap=plt.get_cmap('gray'))
plt.savefig('lensed.png')
plt.imshow(unlensed.array, cmap=plt.get_cmap('gray'))
plt.savefig('unlensed.png')
plt.imshow(diff, cmap=plt.get_cmap('gray'))
plt.savefig('diff.png')
plt.imshow(diff, cmap=plt.get_cmap('gray'), vmin=min(lensed.array), vmax=max(lensed.array))
plt.imshow(diff, cmap=plt.get_cmap('gray'))
plt.vlines
plt.get_cmap
plt.imshow(lensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.show()
plt.imshow(lensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest", vmax=255, vmin=0)
plt.show()
p = plt.imshow(lensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest")
p
p.v
max(lensed.array)
import numpy as np
np.max(lensed.array)
max = _
max
min = np.min(lensed.array)
min
min = 0
p = plt.imshow(lensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest", vmax=max, vmin=min)
p.show()
plt.show()
plt.imshow(diff, cmap=plt.get_cmap('gray'), interpolation="nearest", vmax=max, vmin=min)
plt.show()
plt.imshow(diff, interpolation="nearest", vmax=max, vmin=min)
plt.show()
plt.imshow(diff, interpolation="nearest")
plt.show()
plt.imshow(diff, interpolation="nearest")
plt.colorbar()
plt.show()
plt.imshow(diff, interpolation="nearest")
plt.colorbar()
plt.savefig("diff.png")
p = plt.imshow(lensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.colorbar()
plot.savefig("lensed.png")
plt.savefig("lensed.png")
plt.close()
p = plt.imshow(lensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.colorbar()
plt.savefig('lensed.png')
plt.close()
p = plt.imshow(unlensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.colorbar()
plt.savefig('unlensed.png')
lensed.addNoiseSNR(galsim.GaussianNoise, 50, preserve_flux=True)
lensed.addNoiseSNR(galsim.GaussianNoise(), 50, preserve_flux=True)
plt.close
p = plt.imshow(lensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.colorbar()
plt.savefig('noisy.png')
plt.close()
p = plt.imshow(lensed.array, cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.colorbar()
plt.savefig('noisy.png')
unlensed.addNoiseSNR(galsim.GaussianNoise(), 50, preserve_flux=True)
noisydiff= lensed.array - unlensed.array
plt.close()
plt.imshow(noisydiff,  interpolation="nearest")
plt.colorbar()
plt.savefig('noisydiff.png')
get_ipython().magic(u'save galplots 0-95')
