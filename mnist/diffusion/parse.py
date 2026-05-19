import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--save-path", default="checkpoint.pt")

    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--val-size", type=int, default=1_000)
    parser.add_argument("--test-size", type=int, default=10_000)


    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--inference-steps", type=int, default=50)


    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--muon-lr", type=float, default=1e-3)
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn",
        choices=["original", "match_rms_adamw"],
        default="match_rms_adamw",
    )


    parser.add_argument("--t-dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=0)


    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    
    return parser.parse_args()
