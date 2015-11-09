import utils
import disk_bulge_shear_estimator
import bulge_shear_estimator
sigma = .05

Id = .5
rd = 2

#Ib = .0001
#rb = .5
Ib = .0000001
rb = .5
scale = .1

image, origin = utils.noisy_egg(rd,Id,rb,Ib, disk_g=.3, scale = scale)

Id *= scale**2
rd /= scale 

utils.view_image(image)

#disk_bulge_shear_estimator.estimate_shear(
#    image, x0=center, y0=center, Id=Id, Ib=Ib, ad=rd, ab=rb)
#disk_bulge_shear_estimator.estimate_shear(
#    image, x0=origin.x, y0=origin.y, Id=Id, Ib=0, ad=rd, ab=rb, xid=3)

bulge_shear_estimator.estimate_shear(image, x0=origin.x, y0=origin.y, I=Id, a=rd, b=rd) 
