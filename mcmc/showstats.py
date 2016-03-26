import pickle
import sys

stats = pickle.load(open(sys.argv[1], 'rb'))

for p in stats.iteritems():
    print("%s: %s" % p)
