import model
import platform

SNR = 50
NP = 200
scale = .2
mask = [True] * model.EggParams.nparams

nwalkers = 800
nburnin = 500
nsample = 1000
nthreads = 16 if "cosmos5" in platform.node() else 1


print(nthreads)
