import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-path", default="wgan.pt")
    parser.add_argument("--hf-repo", default=os.environ.get("HF_CHECKPOINT_REPO"))
    parser.add_argument(
        "--hf-repo-type",
        default=os.environ.get("HF_CHECKPOINT_REPO_TYPE", "model"),
        choices=["model", "dataset", "space"],
    )
    parser.add_argument("--hf-prefix", default=os.environ.get("HF_CHECKPOINT_PREFIX"))
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--no-hf-upload", action="store_true")

    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--val-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=10_000)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-dir", default="samples")

    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--leaky-slope", type=float, default=0.2)

    parser.add_argument("--n-critic", type=int, default=1)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--lazy-gp", type=int, default=1)
    parser.add_argument("--recon-weight", type=float, default=0.1)

    parser.add_argument("--optimizer-g", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--optimizer-c", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-c", type=float, default=1e-4)
    parser.add_argument("--beta1-g", type=float, default=0.0)
    parser.add_argument("--beta1-c", type=float, default=0.0)
    parser.add_argument("--beta2-g", type=float, default=0.9)
    parser.add_argument("--beta2-c", type=float, default=0.9)
    parser.add_argument("--weight-decay-g", type=float, default=0.0)
    parser.add_argument("--weight-decay-c", type=float, default=0.0)

    parser.add_argument("--muon-lr-g", type=float, default=1e-3)
    parser.add_argument("--muon-lr-c", type=float, default=1e-3)
    parser.add_argument("--muon-weight-decay-g", type=float, default=0.0)
    parser.add_argument("--muon-weight-decay-c", type=float, default=0.0)
    parser.add_argument("--muon-momentum-g", type=float, default=0.95)
    parser.add_argument("--muon-momentum-c", type=float, default=0.95)
    parser.add_argument("--muon-nesterov-g", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-nesterov-c", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-ns-steps-g", type=int, default=5)
    parser.add_argument("--muon-ns-steps-c", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn-g",
        choices=["original", "match_rms_adamw"],
        default="match_rms_adamw",
    )
    parser.add_argument(
        "--muon-adjust-lr-fn-c",
        choices=["original", "match_rms_adamw"],
        default="match_rms_adamw",
    )

    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")

    return parser.parse_args()
