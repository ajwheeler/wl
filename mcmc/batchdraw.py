import os
import numpy as np
import drawcorner

for dir in os.listdir('.'):
    os.chdir(dir)
    print(os.getcwd())

    for b in ['shear.single', 'shear.double']:
        chain = np.load(b + '.chain.npy')
        kormendy = np.load(b + '.kormendy.npy')
        orientation = np.load(b + '.orientation.npy')

        with open(b + '.stats.p', 'rb') as f:
            stats = pickle.load(f)

        print(stats['mask'])

        fig = drawcorner.make_figure(chain,
                                     stats['fiducial_params'].toArray(stats['mask']),
                                     mask=stats['mask'])
        fig.savefig(b + '.png')

        fig = drawcorner.make_figure(chain,
                                     stats['fiducial_params'].toArray(stats['mask']),
                                     weights=kormendy*orientation,
                                     mask=stats['mask'])
        fig.savefig(b + '.withpriors.png')

    os.chdir('..')
