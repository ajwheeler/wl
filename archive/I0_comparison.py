import centroid_regression
import utils
import detectability
import I0_estimator

flux = 120
scale_radius = 5
sigma = .05

#set this low enough that find_galaxy will always hit
threshold = .4

image = utils.noisy_exp(scale_radius, flux, sigma=sigma).array
smoothed_image = detectability.convolve_with_model(image,scale_radius)
centroid = detectability.find_galaxy(smoothed_image, threshold)

print("estimated centroid: " + str(centroid))
print("I in smoothed image at estimated centroid: " + str(smoothed_image[centroid]))
print("I in smoothed image at true centroid: " + str(smoothed_image[(42.5, 42.5)]))

I = I0_estimator.estimated_I0(image, (42.5, 42.5), 5)
print("Specialized I0 estimator estimate: " + str(I))

params = centroid_regression.regress(image, interactive=False, iterations=40, output=False)
print("centroid regression for unsmoothed image:")
print("A = %s, r0 = %s, I0 = %s" % params)
