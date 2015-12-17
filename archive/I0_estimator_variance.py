import utils
from estimators import estimated_I0

flux = 100
scale_radius = 5
sigma = .05
iterations = 500

#WARNING: changing the flux or the scale radius necessitates changing this
A = (42.5, 42.5)

I0s = []
for i in xrange(iterations):
    data = utils.noisy_exp(scale_radius, flux, sigma=sigma).array
    I0 = estimated_I0(data, A, scale_radius)
    I0s.append(I0)

data = utils.noisy_exp(scale_radius, flux, scale=1, sigma=sigma).array
print(len(data))
analytic_var = utils.analytic_I0_variance(data, A, scale_radius, sigma)

print("analytically derived variance: " + str(analytic_var))

true_I0 = utils.flux2I0(flux, scale_radius)
mean = sum(I0s) / len(I0s)
variance = sum([(I - mean)**2 for I in I0s])/len(I0s)
true_variance = sum([(I - true_I0)**2 for I in I0s])/len(I0s)


print("mean I0: " + str(mean))
print("true I0: " + str(true_I0))
print("variance: " + str(variance))
print("true_variance: " + str(variance))
