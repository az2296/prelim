# prelim

Conditional diffusion and conditional WGAN-GP experiments on the toy data
models `M1`–`M4` and on MNIST.

Trained checkpoints are mirrored to a public Hugging Face repo:
https://huggingface.co/az2296/prelim-checkpoints

## Requirements

| Package      | Version |
| ------------ | ------- |
| Python       | 3.14.5  |
| PyTorch      | 2.12.0  |
| torchvision  | 0.27.0  |
| diffusers    | 0.38.0  |
| numpy        | 2.4.6   |
| matplotlib   | 3.10.9  |
| pandas       | 3.0.3   |

These are the versions used to produce the results in this repo. PyTorch
must include `torch.optim.Muon`; we tested on 2.12.0, but earlier 2.x
releases that ship it should also work. `diffusers` is only used for
`DDIMScheduler`; any 0.30+ release should be fine.

Training defaults to `--device mps`. `cuda` and `cpu` are also supported.

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

## Download checkpoints

To reproduce the notebooks without retraining, download all checkpoints
from the public Hugging Face repo into their matching local paths:

```bash
python scripts/download_checkpoints.py
```

The script uses stdlib only, skips files that already exist, and accepts
`--filter m3m4` / `--filter mnist` to restrict the download, plus
`--dry-run` and `--force`.

## Uploading new checkpoints (optional)

`scripts/hf_checkpoints.py` can push freshly trained checkpoints back to
the Hub. It requires `huggingface_hub`:

```bash
pip install huggingface_hub
```

If `huggingface_hub` is not installed, training and the download script
work fine — uploads simply skip with a warning.
