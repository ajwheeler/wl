'''truncate chain and lnprob files on the command line'''
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser(description="truncate chain and lnprob files")
parser.add_argument('file_prefix', type=str)
parser.add_argument('-b', '--nburnin', type=int, required=True)
args = parser.parse_args()

with open(args.file_prefix + '.stats.p', 'rb') as f:
    stats = pickle.load(f)

#Load files
chain = np.load(args.file_prefix + '.chain.npy')
lnprob = np.load(args.file_prefix + '.lnprob.npy')

#reinflate chain and lnprob
dim = sum(stats['mask'])
chain = chain.reshape((stats['nwalkers'], stats['nsample'], dim))
lnprob = lnprob.reshape((stats['nwalkers'], stats['nsample']))

#remove first nburnin from arrays, reshape
chain = np.delete(chain, np.s_[:args.nburnin],1)
chain = chain.reshape((stats['nwalkers'] * (stats['nsample']-args.nburnin), dim))
lnprob = np.delete(lnprob, np.s_[:args.nburnin],1).flatten()

#update stats
stats['nburnin'] += args.nburnin
stats['nsample'] -= args.nburnin

#construct new prefix
new_prefix = args.file_prefix.split('.')
new_prefix[2] = str(stats['nburnin'])
new_prefix[3] = str(stats['nsample'])
new_prefix = '.'.join(new_prefix)

#save files
statsfn = new_prefix + '.stats.p'
print("writing " + statsfn)
with open(statsfn, 'wb') as f:
    pickle.dump(stats,f)

chainfn = new_prefix + '.chain.npy'
print('writing ' + chainfn)
np.save(chainfn, chain)

lnfn = new_prefix + '.lnprob.npy'
print('writing ' + lnfn)
np.save(lnfn, lnprob)
