# prelim

Public checkpoints are stored on Hugging Face:

https://huggingface.co/az2296/prelim-checkpoints

## Requirements

The exact versions used for the results in this repo:

| Package          | Version |
| ---------------- | ------- |
| Python           | 3.14.5  |
| PyTorch          | 2.12.0  |
| torchvision      | 0.27.0  |
| diffusers        | 0.38.0  |
| numpy            | 2.4.6   |
| matplotlib       | 3.10.9  |
| pandas           | 3.0.3   |

Notes:

- **PyTorch must include `torch.optim.Muon`.** The training scripts construct
  Muon directly from `torch.optim`, so a PyTorch old enough to predate the
  upstream Muon will fail at optimizer construction whenever a `muon`
  optimizer flag is passed. We tested on `torch==2.12.0`; earlier 2.x
  releases that ship `torch.optim.Muon` should also work.
- **`diffusers`** is only used for `DDIMScheduler` in the diffusion training
  loops; any 0.30+ release should work, but 0.38.0 is what we ran.
- **`huggingface_hub`** is *not* required to reproduce the results — it is
  only used by `scripts/hf_checkpoints.py` to **upload** new checkpoints to
  the Hub. Downloading existing checkpoints (see below) uses the public
  HTTPS endpoint and needs nothing extra. Install `huggingface_hub` only if
  you plan to push your own checkpoints.
- **Device.** Training defaults to `--device mps` (Apple Silicon). `cuda` and
  `cpu` are also supported.

## Install

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install \
    "torch==2.12.0" \
    "torchvision==0.27.0" \
    "diffusers==0.38.0" \
    "numpy==2.4.6" \
    "matplotlib==3.10.9" \
    "pandas==3.0.3" \
    jupyter
```

## Checkpoints

All trained checkpoints used in the notebooks are mirrored to the public
Hugging Face repo above. To download them into the matching local paths
(e.g. `m3m4/diffusion/notebook_ckpts/...`):

```bash
python scripts/download_checkpoints.py
```

The script uses stdlib only (no `huggingface_hub` install needed), skips
files that already exist, and accepts `--filter m3m4` / `--filter mnist`
to restrict the download, plus `--dry-run` and `--force`.
