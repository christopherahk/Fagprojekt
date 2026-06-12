import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os

SEQ_LEN = 16
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.002


class CNN1D(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


def prepare_dataset(rms_data, angles_ds):
    rms_data = rms_data[:56, :]
    X = rms_data.T

    y = np.zeros(len(angles_ds), dtype=np.int64)
    y[angles_ds > 2.0] = 1
    y[angles_ds < -2.0] = 2

    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(X)):
        X_seq.append(X[i - SEQ_LEN:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def load_rat_file(path):
    data = np.load(path)
    return data["rms_data"], data["a"], data["split_idx"]


def split_data(X, y, train=0.70, val=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    train_end = int(train * len(idx))
    val_end = int((train + val) * len(idx))

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def train_model(X_train, y_train, X_val, y_val):
    n_channels = X_train.shape[-1]

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train.reshape(-1, n_channels))
    X_val_2d = scaler.transform(X_val.reshape(-1, n_channels))

    X_train_t = torch.tensor(
        X_train_2d.reshape(X_train.shape), dtype=torch.float32
    ).permute(0, 2, 1)

    X_val_t = torch.tensor(
        X_val_2d.reshape(X_val.shape), dtype=torch.float32
    ).permute(0, 2, 1)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    device = torch.device("cpu")
    model = CNN1D(n_channels=n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = criterion(val_logits, y_val_t.to(device)).item()
            val_preds = val_logits.argmax(dim=1).cpu()
            val_acc = (val_preds == y_val_t).float().mean().item()

        avg_train_loss = train_loss / len(y_train)
        print(f"  Epoch {epoch+1}/{EPOCHS} — train_loss: {avg_train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t.to(device)).argmax(dim=1).cpu().numpy()

    return model, y_val, val_preds


def leave_one_rat_out(data_dir, rat_ids, seed=42):
    results = {}

    for test_rat in rat_ids:
        print(f"Testing on RAT {test_rat}")

        X_train_list, y_train_list = [], []
        X_test, y_test = None, None

        for rat in rat_ids:
            path = os.path.join(data_dir, f"rat{rat}.npz")
            rms, angles, split_idx = load_rat_file(path)
            X, y = prepare_dataset(rms, angles)

            if rat == test_rat:
                X_test, y_test = X, y
            else:
                X_train_list.append(X)
                y_train_list.append(y)

        X_all = np.concatenate(X_train_list)
        y_all = np.concatenate(y_train_list)

        X_train, y_train, X_val, y_val, X_held, y_held = split_data(
            X_all, y_all, train=0.70, val=0.15, seed=seed
        )

        print(f"Train: {X_train.shape} | Val: {X_val.shape} | Held-out: {X_held.shape} | Test rat: {X_test.shape}")

        model, y_true, y_pred = train_model(X_train, y_train, X_val, y_val)

        print(classification_report(y_true, y_pred, target_names=["Neutral", "Dorsi", "Plantar"]))
        results[test_rat] = classification_report(y_true, y_pred, output_dict=True)

    return results


if __name__ == "__main__":
    data_dir = "./dataset_rats_test"
    rat_ids = list(range(4, 11))

    results = leave_one_rat_out(data_dir, rat_ids)
