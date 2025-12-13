import os
import subprocess

UNIFORMITY_WEIGHTS = [0.0, 0.5, 1.0]


def run_experiment(weight):
    save_prefix = f"contrastive_learning/u{weight}".replace(".", "")
    cmd = [
        "python",
        "contrastive_learning/train_with_split.py",
        "--dim",
        "256",
        "--epochs",
        "100",
        "--batch",
        "512",
        "--tau",
        "0.07",
        "--lr",
        "1e-3",
        "--eval-every",
        "10",
        "--uniformity-weight",
        str(weight),
        "--save-prefix",
        save_prefix,
    ]

    print(f"\n========== Running uniformity_weight={weight} ==========")
    print(" ".join(cmd))
    subprocess.run(cmd)


def main():
    for w in UNIFORMITY_WEIGHTS:
        run_experiment(w)

    print("\n===========================================")
    print("All experiments completed!")
    print("Saved under contrastive_learning/u0, u05, u1")
    print("===========================================\n")


if __name__ == "__main__":
    main()
