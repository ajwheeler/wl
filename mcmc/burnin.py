import numpy as np
import sys

if __name__ == '__main__':
    inputf = sys.argv[1]
    nburnin = int(sys.argv[2])

    chain = np.load(inputf)

    inputf = inputf.split('.')
    inputf[-4] = str(nburnin)
    inputf = '.'.join(inputf)

    print("saving new chain to "  + inputf)    
    np.save(inputf, chain[nburnin:])



