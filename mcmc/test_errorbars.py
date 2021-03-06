"""run several 1D chains, and test that the error bars are consistent with the scatter"""
import matplotlib
matplotlib.use('TkAgg')
import mcmc
import model
import calculate_mode
import numpy as np

nwalkers = 100
nsample = 200
nburnin = 20

mask = [True] + [False]*10
dual_band = False
trueParams = model.EggParams()


trials = 20
qs = []
for _ in range(trials):
    data, var = mcmc.generate_data(trueParams, dual_band)

    sampler = mcmc.run_chain(data, var, trueParams, nwalkers, nburnin,
                             nsample, mask=mask, dual_band=dual_band,
                             collect_stats=False)
    # print("autocorrelation time: {}".format(stats['autocorrelation_time']))
    # print("acceptance fraction: {}".format(stats["acceptance_fraction"]))
    q = calculate_mode.median(sampler.flatchain)[0]
    qs.append(q)
    print("{} (+{}) (-{})".format(q[1], q[1]-q[0], q[2]-q[1]))
    print("")

qs = np.array(qs)

print(qs)

q = sum(qs)/trials
print("{} (+{}) (-{})".format(q[1], q[1]-q[0], q[2]-q[1]))

means = qs[:,1]
print("scatter: " + str(np.std(means)))

# import drawcorner
# import matplotlib.pyplot as plt
# drawcorner.make_figure(sampler.flatchain, [3.0])
# plt.show()
