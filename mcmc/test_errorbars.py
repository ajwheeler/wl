"""run several 1D chains, and test that the error bars are consistent with the scatter"""

import mcmc
import model
import calculate_mode

nwalkers = 100
nsample = 200
nburnin = 30

mask = [True] + [False]*10
dual_band = False
trueParams = model.EggParams()

data, var = mcmc.generate_data(trueParams, dual_band)

sampler, stats = mcmc.run_chain(data, var, trueParams, nwalkers, nburnin,
                                nsample, mask=mask, dual_band=dual_band)
print(calculate_mode.median(sampler.flatchain()))
