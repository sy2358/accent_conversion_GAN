import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import imageio
import argparse
import numpy as np
import os
import sys
import math

cmap = plt.get_cmap('jet')

import audio_utilities

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str, required=True,
                    help='Input WAV file')
parser.add_argument('--out_file_prefix', type=str,
                    help='Output file prefix (with extension)')
parser.add_argument('--out_file_dir', type=str,
                    help='Output file dir')
parser.add_argument('--sample_rate_hz', default=44100, type=int,
                    help='Sample rate in Hz')
parser.add_argument('--fft_size', default=2048, type=int,
                    help='FFT siz')
parser.add_argument('--min_freq_hz', default=70, type=int,
                    help='Minimal frequency (Herz)')
parser.add_argument('--max_freq_hz', default=8000, type=int,
                    help='Maximal frequency (Herz)')
parser.add_argument('--mel_bin_count', default=256, type=int,
                    help='Count of MEL bins')
parser.add_argument('--overlap_ratio', default=8, type=int,
                    help='fft window overlap 1/N ratio')
parser.add_argument('--pad_length', type=int,
                    help='if defined pad up to this length (in frame) or if longer, fail')
args = parser.parse_args()

in_file = args.in_file
out_file_prefix = args.out_file_prefix
if out_file_prefix is None or out_file_prefix.endswith(".wav"):
    out_file_prefix = in_file[:-4]

if args.out_file_dir is not None:
    out_file_prefix = os.path.join(args.out_file_dir, out_file_prefix)

# Load an audio file. It must be WAV format. Multi-channel files will be
# converted to mono.
input_signal = audio_utilities.get_signal(in_file, expected_fs=args.sample_rate_hz)

print("processing %s - signal %fs long" % (in_file, len(input_signal)*1.0/args.sample_rate_hz))
if args.pad_length:
    if len(input_signal) > args.pad_length:
        print("signal too long... skipping")
        sys.exit(0)
    else:
        max=np.amax(input_signal)
        whitenoise = np.random.normal(0, max/100, size=args.pad_length-len(input_signal))
        input_signal = np.concatenate((input_signal, whitenoise))

# Hopsamp is the number of samples that the analysis window is shifted after
# computing the FFT. For example, if the sample rate is 44100 Hz and hopsamp is
# 256, then there will be approximately 44100/256 = 172 FFTs computed per second
# and thus 172 spectral slices (i.e., columns) per second in the spectrogram.
hopsamp = args.fft_size // args.overlap_ratio

# Compute the Short-Time Fourier Transform (STFT) from the audio file. This is a 2-dim Numpy array with
# time_slices rows and frequency_bins columns. Thus, you will need to take the
# transpose of this matrix to get the usual STFT which has frequency bins as rows
# and time slices as columns.
stft_full = audio_utilities.stft_for_reconstruction(input_signal,
                                                    args.fft_size, hopsamp)

# Note that the STFT is complex-valued. Therefore, to get the (magnitude)
# spectrogram, we need to take the absolute value.
stft_mag = np.abs(stft_full)**2

linear_bin_count = 1 + args.fft_size//2
filterbank = audio_utilities.make_mel_filterbank(args.min_freq_hz, args.max_freq_hz, args.mel_bin_count,
                                                 linear_bin_count , args.sample_rate_hz)

mel_spectrogram = np.dot(filterbank, stft_mag.T)

# convert to decibels
mel_spectrogram = np.abs(20.*np.log10(mel_spectrogram/10e-6))

scale_mel = 1.0 / np.amax(mel_spectrogram)
# Rescale to put all values in the range [0, 1].
mel_spectrogram *= scale_mel
mel_spectrogram = np.flip(mel_spectrogram, 0)

print('==>',out_file_prefix)
imageio.imwrite(out_file_prefix+"_spectogram_bw.png", (65535*(1-mel_spectrogram)).astype(np.uint16))
imageio.imwrite(out_file_prefix+"_spectogram_color.png", (255*(cmap(mel_spectrogram))).astype(np.uint8)[:,:,:3])
with open(out_file_prefix+"_params.txt", "wt") as f:
    f.write("%f\n" % scale_mel)
    f.write("%d\n" % hopsamp)
    f.write("%d\n" % args.min_freq_hz)
    f.write("%d\n" % args.max_freq_hz)
    f.write("%d\n" % args.mel_bin_count)
    f.write("%d\n" % args.fft_size)
    f.write("%d\n" % args.sample_rate_hz)
    f.write("%s\n" % args.in_file)

print("...complete")
