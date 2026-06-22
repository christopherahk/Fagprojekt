import sys
import matplotlib.pyplot as plt


def parse_results(path):
    val_loss, val_acc = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("VAL_LOSS"):
                val_loss.append(float(line.split(":")[1]))
            elif line.startswith("VAL_ACC"):
                val_acc.append(float(line.split(":")[1]))
    return val_loss, val_acc


def plot_results(val_loss, val_acc, out_path="val_results.png"):
    epochs = range(1, len(val_loss) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color1 = "tab:red"
    ax1.set_xlabel("Validation step")
    ax1.set_ylabel("Validation Loss", color=color1)
    ax1.plot(epochs, val_loss, color=color1, marker="o", markersize=3, label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Validation Accuracy", color=color2)
    ax2.plot(epochs, val_acc, color=color2, marker="s", markersize=3, label="Val Acc")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    plt.title("Validation Loss & Accuracy")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "results_val.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "val_results.png"
    loss, acc = parse_results(in_path)
    print(f"Parsed {len(loss)} loss points and {len(acc)} acc points")
    plot_results(loss, acc, out_path)