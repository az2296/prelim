import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model on M3/M4.")

    parser.add_argument("--model", choices=["M3", "M4"], default="M3")
    parser.add_argument("--t-dim", type=int, default=64)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--end-seed", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--hf-repo", default=os.environ.get("HF_CHECKPOINT_REPO"))
    parser.add_argument(
        "--hf-repo-type",
        default=os.environ.get("HF_CHECKPOINT_REPO_TYPE", "model"),
        choices=["model", "dataset", "space"],
    )
    parser.add_argument("--hf-prefix", default=os.environ.get("HF_CHECKPOINT_PREFIX"))
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--no-hf-upload", action="store_true")
    parser.add_argument("--arch", choices=["mlp", "film", "deep"], default="mlp")
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--num-train", type=int, default=5000)
    parser.add_argument("--num-val", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=100)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adamw", "muon", "normuon"], default="muon")
    parser.add_argument("--muon-lr", type=float, default=1e-3)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adjust-lr-fn", choices=["original", "match_rms_adamw"], default="match_rms_adamw")
    parser.add_argument("--normuon-beta2", type=float, default=0.95)

    parser.add_argument("--num-train-timesteps", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--j-val", type=int, default=20)
    parser.add_argument("--j-test", type=int, default=1000)

    return parser.parse_args()
