import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image', type=str, help='path of image')

args = parser.parse_args()

if args.image[-4:] == '.png' or args.image[-4:] == '.jpg':
    img = plt.imread(args.image)
elif args.image[-4:] == '.npy':
    img = np.load(args.image)

mean = np.load("mean_image.npz")
mean = mean.get(mean.files[0])

img = img - mean
img = img / 128

np.save(args.image[:-4] + '_preprocessed', img)