import argparse
import os
import os.path
import numpy as np

parser = argparse.ArgumentParser(prog='fit',
                                 description='Fit beta1 and gamma1 using new symmetry approach!')

parser.add_argument('--files', nargs='*', required=True, help='npy files to combine')
parser.add_argument('--out', required=True, help='npy output filename')

args = parser.parse_args()

npys = []
for file in args.files:
    print(file)
    thisnpy = np.load(file)
    print(thisnpy.shape)
    npys.append(thisnpy)

np.save(args.out, np.concatenate(npys))
