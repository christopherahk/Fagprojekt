import matplotlib.pyplot as plt

pretrain_file = "./results_val_pretrain.txt"
no_pretrain_file = "./results_val_no_pretrain.txt"

files = [pretrain_file, no_pretrain_file]

val_losses_pretrain = []
val_accs_pretrain = []
val_losses_no_pretrain = []
val_accs_no_pretrain = []
val_losses = [val_losses_pretrain, val_losses_no_pretrain]
val_accs = [val_accs_pretrain, val_accs_no_pretrain]

for file in files:
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            if "VAL_LOSS" in line:
                val_loss = float(line.split(":")[1].strip())
                if file == pretrain_file:
                    val_losses_pretrain.append(val_loss)
                elif file == no_pretrain_file:
                    val_losses_no_pretrain.append(val_loss)
            if "VAL_ACC" in line:
                val_acc = float(line.split(":")[1].strip())
                if file == pretrain_file:
                    val_accs_pretrain.append(val_acc)
                elif file == no_pretrain_file:
                    val_accs_no_pretrain.append(val_acc)


epochs_pretrain = range(1, len(val_losses[0]) + 1)
epochs_no_pretrain = range(1, len(val_losses[1]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs_pretrain, val_losses[0], marker='o', label="With pre-training")
ax1.plot(epochs_no_pretrain, val_losses[1], marker='o', color="orange", label="Without pre-training")
ax1.set_title("Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(epochs_pretrain, val_accs[0], marker='o', label="With pre-training")
ax2.plot(epochs_no_pretrain, val_accs[1], marker='o', color="orange", label="Without pre-training")
ax2.set_title("Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("./outputs/plotted_val.png")
plt.show()
