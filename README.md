# FuseFi: Combining Irregularly Sampled CSI from Diverse Communication Packets and Frequency Bands for Wi-Fi Sensing

Reference implementation of the time-aware DNN from **"UniFi: Combining
Irregularly Sampled CSI from Diverse Communication Packets and Frequency Bands
for Wi-Fi Sensing"** — Gaofeng Dong, Kang Yang, Mani Srivastava (ECE, UCLA).

Paper: <https://arxiv.org/abs/2512.22143>

## Overview

Existing Wi-Fi sensing systems inject high-rate probing packets (100–1000 Hz) to
obtain uniformly sampled Channel State Information (CSI), costing 40%+ of
communication throughput. FuseFi instead learns directly from the **irregularly
sampled CSI that already exists in normal communication traffic** — across data,
management and control frames on both the 2.4 and 5 GHz bands — with no packet
injection.

FuseFi has two stages:

1. **CSI sanitization** (offline, MATLAB): clusters packets by PHY format,
   normalizes amplitude, aligns waveforms, removes bursty packets and optionally
   selects subcarriers. Its output is the `.mat` dataset consumed here.
2. **Time-aware DNN** (this repo, PyTorch):
   learns from non-uniform timestamps without resampling. It fuses CSI and time
   encodings into content-aware attention keys and drops the missingness mask as
   an input feature (CSI missingness is not motion-related). A GRU + MLP head
   produces the activity prediction.


## Install

```bash
conda env create -f environment.yml && conda activate FuseFi
# or, into an existing environment:
pip install -r requirements.txt
```

Tested with Python 3.12.1, PyTorch 2.2.1, CUDA 11.8.


## Usage


```bash
cd src
python tanenc_classification.py
```

Override only what you want to change:

```bash
# a quick smoke test: one seed, a few epochs
python tanenc_classification.py --niters 5 --seed-len 1
```



## Repository layout

```
src/tanenc_classification.py   train/eval loop (entry point)
src/models.py                  enc_mtan_classif_csi = FuseFi DNN,
                               multiTimeAttention, CombineEmbed
src/utils.py                   dataloaders, burst filtering / union-timeline
                               batching, normalization, evaluate_classifier
src/testbed_dataset.py         Testbed_Dataset_IrregularNSS (CommCSI-HAR reader)
```



## Citation

```bibtex
@article{dong2025unifi,
  title   = {UniFi: Combining Irregularly Sampled CSI from Diverse Communication
             Packets and Frequency Bands for Wi-Fi Sensing},
  author  = {Dong, Gaofeng and Yang, Kang and Srivastava, Mani},
  journal = {arXiv preprint arXiv:2512.22143},
  year    = {2025}
}
```

## Acknowledgements

Built on [mTAN](https://github.com/reml-lab/mTAN) (Shukla & Marlin, ICLR 2021).
