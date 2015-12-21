import numerical_elipticity_estimator as nee
import numpy as np
import utils

params = nee.EggParams()
egg = nee.noisy_egg(params)

dmdgamma = nee.dmd(egg, params, 'g1s')

print(np.sum(dmdgamma**2))
utils.view_image(dmdgamma)

dmdIA = nee.dmd(egg, params, 'g1d')
utils.view_image(dmdIA)

dmdgammaP = dmdgamma - (np.sum(dmdgamma * dmdIA)/\
                        np.sqrt(np.sum(dmdIA**2)* np.sum(dmdgamma**2)))*dmdIA


utils.view_image(dmdgammaP)

