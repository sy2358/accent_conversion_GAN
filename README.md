<br><br><br>

# CycleGAN in PyTorch for Spectrogram Style Transfer and Speech Generation

We provide PyTorch implementations for both unpaired and paired image-to-image translation.

The code was written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung89), and supported by [Tongzhou Wang](https://ssnl.github.io/).

This PyTorch implementation produces results comparable to or better than our original Torch software. If you would like to reproduce the same results as in the papers, check out the original [CycleGAN Torch](https://github.com/junyanz/CycleGAN) and [pix2pix Torch](https://github.com/phillipi/pix2pix) code

**Note**: The current software works well with PyTorch 0.4+. Check out the older [branch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/pytorch0.3.1) that supports PyTorch 0.1-0.3.



**Speech Style Transfer using CycleGAN and Pix2Pix: [Paper](https://arxiv.org/ftp/arxiv/papers/1904/1904.09407.pdf)**
<img src="https://github.com/sy2358/pytorch-CycleGAN-and-pix2pix/blob/master/imgs/compare_spectrograms.jpg" width="800"/>




If you use this code for your research, please cite:

Self-imitating Feedback Generation Using GAN for Computer-Assisted Pronunciation Training. Seung Hee Yang and Minhwa Chung (2019) https://arxiv.org/abs/1904.09407. (To be published in Proceedings of INTERSPEECH 2019, Graz, Austria).

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)
In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)

Griffin D. and Lim J. (1984). "Signal Estimation from Modified Short-Time Fourier Transform". IEEE Transactions on Acoustics, Speech and Signal Processing. 32 (2): 236–243. doi:10.1109/TASSP.1984.1164317

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch 0.4+ and torchvision from http://pytorch.org and other dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/sy2358/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```
- For Conda users, we include a script `./scripts/conda_deps.sh` to install PyTorch and other libraries.

### 1. Speech2Spectrogram Conversion

The provided code shows an example usage of the Griffin and Lim algorithm. It loads an audio file, computes the spectrogram, optionally performs low-pass filtering by zeroing all frequency bins above some cutoff frequency, and then uses the Griffin and Lim algorithm to reconstruct an audio signal from the modified spectrogram. Finally, both the reconstructed audio signal and the spectrogram plot figure are saved to a file.

```bash
python3 build-melspec-from-wav.py --in_file ../data/CN221s3_219.wav --sample_rate_hz 16000 --fft_size 512 --overlap_ratio 3 --mel_bin_count 128 --max_freq_hz 5000 --pad_length 24000
```
It generates 128x128 gray and colour spectrogram images. 

### 2. CycleGAN train/test

- Train a model:
```bash
python3 train.py —dataroot data/CycleGan_data —name cyclegan —model cycle_gan —no_flip —resize_or_crop none —loadSize 128 —fineSize 128
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`

see results on the training data in the following directory:
~/pytorch-CycleGAN-and-pix2pix/checkpoints/cyclegan/web/images

- Test the model:
```bash
python3 test.py —dataroot data/CycleGan_data/ —name cyclegan —model cycle_gan —no_flip —loadSize 256 —fineSize 256 —num_test 200 —no_dropout —results_dir test_results/
```

## [Datasets](docs/datasets.md)
Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)
Before you post a new question, please first look at the above Q & A and existing GitHub issues.


### 3. Resize the Generated Images to 128x128 for Conversion back to Sound
```bash
python3 ..resize.py --in_file results/cyclegan/test_latest/images/9_real_A.png --resize_file results/cyclegan/test_latest/images/9_real_A_resized.png --resize 128 128
```

### 4. Rebuild the Waveform
```bash
build-wav-from-melspec.py --in_file results/cyclegan/test_latest/images/9_real_A_resized.png --param_file ~/test/9_params.txt --out_file results/cyclegan/test_latest/images/9_real_A-rebuild.wav --iterations 1000
```


## Citation
If you use this code for your research, please cite the following papers.
```

@inproceedings{Yang2019,
  title={Self-imitating Feedback Generation Using GAN for Computer-Assisted Pronunciation Training},
  author={Yang, Seung Hee, and Chung, Minhwa},
  booktitle={Interspeech, 2019 IEEE International Conference on},
  year={2019}
}

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

```


## Acknowledgments
Our code comes from [CycleGAN](https://github.com/junyanz/CycleGAN) and [Pix2Pix](https://github.com/phillipi/pix2pix), which are inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan), and from [Griffin_Lim](https://github.com/bkvogel/griffin_lim).
