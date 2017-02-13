import argparse
import pickle
import matplotlib.pyplot as plt
from mcmc import QuietImage

parser = argparse.ArgumentParser(
    description='draw gal image from pickled file')
parser.add_argument('data_file')
args = parser.parse_args()

if args.data_file[-2:] != ".p":
    raise RuntimeError("input file must be a .p")
output_filename = args.data_file[:-2] + ".png"

with open(args.data_file,'r') as f:
    data = pickle.load(f)[-1]
plt.imshow(data.array, cmap=plt.get_cmap('gray'))
print("saving output to " + output_filename)
plt.savefig(output_filename)
