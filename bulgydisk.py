import utils
import disk_bulge_shear_estimator
sigma = .05

Id = .5
rd = 2

Ib = .0001
rb = .5

image = utils.noisy_egg(rd,Id,rb,Ib)

utils.view_image(image)

#center = len(image)/2

#disk_bulge_shear_estimator.estimate_shear(
#    image, x0=center, y0=center, Id=Id, Ib=Ib, rd=rd, rb=rb, n=2, ad=1.5, bd=.5)
