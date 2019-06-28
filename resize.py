import imageio
import argparse
import numpy as np
from skimage.transform import resize

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str, required=True,
                    help='has to be bw')
parser.add_argument('--resize_file', type=str, required=True,
                    help='file resized and grayed')
parser.add_argument('--resize', nargs='+',
                    help='resize parameters (2d)')
args = parser.parse_args()

in_file = args.in_file

spectrogram = imageio.imread(in_file)
args.resize = [int(d) for d in args.resize]
spectrogram = resize(spectrogram, args.resize, mode='constant', anti_aliasing=True)

mel_spectrogram=(1-rgb2gray(spectrogram))

imageio.imwrite(args.resize_file, (65535*(1-mel_spectrogram)).astype(np.uint16))

