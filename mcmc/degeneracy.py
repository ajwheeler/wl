import model
import numpy as np

orientation_params = model.EggParams(g1d = .5, g1b = .5)
lensing_params = model.EggParams(g1s = .5)

oegg = model.egg(orientation_params, dual_band=False)
legg = model.egg(lensing_params, dual_band=False)

print(np.argwhere(oegg.array - legg.array))

model.show(oegg)
model.show(legg)
model.show(oegg-legg)
