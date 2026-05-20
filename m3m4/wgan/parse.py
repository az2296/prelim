import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional WGAN on M3/M4.")

    parser.add_argument("--model", choices=["M3", "M4"], default="M3")
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--end-seed", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--arch", choices=["mlp", "film"], default="mlp")

    parser.add_argument("--num-train", type=int, default=5000)
    parser.add_argument("--num-val", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=100)

    parser.add_argument("--lr-g", type=float, default=1e-3)
    parser.add_argument("--lr-c", type=float, default=1e-3)
    parser.add_argument("--beta1-g", type=float, default=0.0)
    parser.add_argument("--beta2-g", type=float, default=0.9)
    parser.add_argument("--beta1-c", type=float, default=0.0)
    parser.add_argument("--beta2-c", type=float, default=0.9)
    parser.add_argument("--weight-decay-g", type=float, default=0.0)
    parser.add_argument("--weight-decay-c", type=float, default=0.0)
    parser.add_argument("--lr-decay-g", type = float, default=0.995)
    parser.add_argument("--lr-decay-c", type = float, default = 0.995)

    parser.add_argument("--optimizer-g", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--optimizer-c", choices=["adamw", "muon"], default="adamw")
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


    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--recon-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--leaky-slope", type=float, default=0.01)

    parser.add_argument("--j-train",  type=int, default=5)
    parser.add_argument("--j-eval", type=int, default=100)
    parser.add_argument("--j-test", type=int, default=1000)


    args = parser.parse_args()

    return args
