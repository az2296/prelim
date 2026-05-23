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
| huggingface_hub  | 1.16.1  |

Notes:

- **PyTorch must include `torch.optim.Muon`.** The training scripts construct
  Muon directly from `torch.optim`, so a PyTorch old enough to predate the
  upstream Muon will fail at optimizer construction whenever a `muon`
  optimizer flag is passed. We tested on `torch==2.12.0`; earlier 2.x
  releases that ship `torch.optim.Muon` should also work.
- **`diffusers`** is only used for `DDIMScheduler` in the diffusion training
  loops; any 0.30+ release should work, but 0.38.0 is what we ran.
- **`huggingface_hub`** is only used by the optional checkpoint uploader in
  `scripts/hf_checkpoints.py`. If you do not set `HF_CHECKPOINT_REPO` it is
  not invoked at training time, but it is still imported, so keep it
  installed.
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
    "huggingface_hub==1.16.1" \
    jupyter
```