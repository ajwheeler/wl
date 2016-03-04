import model

SNR = 50
NP = 200
scale = .2
mask = [True] * model.EggParams.nparams

nwalkers = 800
nburnin = 500
nsample = 1000

import platform
print(platform.node())
