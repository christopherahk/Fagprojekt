import numpy as np
import os

DATA_DIR = "./splits_6rats"
FILES_TO_CHECK = ["test.npz","train.npz", "val.npz"]

for filename in FILES_TO_CHECK:
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        data = np.load(path)
        labels = data["y"]

        unique, counts = np.unique(labels, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        total = len(labels)
        print(f"\nFile: {filename} (Total: {total})")


        class_names = {0: "Dorsi", 1: "Plantar", 2: "Neutral"}
        for c_id, name in class_names.items():
            count = counts_dict.get(c_id, 0)
            pct = (count / total) * 100
            print(f"  {name:<8}: {count:>6} samples ({pct:>5.2f}%)")
    else:
        pass
