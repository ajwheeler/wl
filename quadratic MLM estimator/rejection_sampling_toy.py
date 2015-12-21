import numpy as np

accepted = []
allPoints = []
for _ in xrange(100000):
    x,y = np.random.random()*2-1, np.random.random()*2-1
    s = np.sqrt(x**2 + y**2)

    r = np.random.random()

    if r < np.exp(-s**2):
        accepted.append((x,y))
    allPoints.append((x,y))

points_cov = np.cov(np.transpose(allPoints))
points_fisher = np.linalg.inv(points_cov)
print("points_fisher")
print(points_fisher)


print(str(len(accepted)) + " points accepted")
covariance_with_prior = np.cov(np.transpose(accepted))
fisher_with_prior = np.linalg.inv(covariance_with_prior)
print("model + prior fisher")
print(fisher_with_prior)

print("true prior fisher")
print(fisher_with_prior - points_fisher)
