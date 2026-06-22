import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os

SEQ_LEN = 16
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
SUBSAMPLE_RATE = 1

RAT_ID = 10


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
    for i in range(SEQ_LEN, len(X), SUBSAMPLE_RATE):
        X_seq.append(X[i - SEQ_LEN:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def load_rat_file(path):
    data = np.load(path)
    return data["rms_data"], data["a"], data["split_idx"]


def split_data_temporal(X, y, train=0.70, val=0.15):
    n = len(X)
    train_end = int(train * n)
    val_end = int((train + val) * n)
    return (
        X[:train_end],        y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:],          y[val_end:],
    )


def to_tensors(X, y, scaler, n_channels, fit_scaler=False):
    if fit_scaler:
        X_2d = scaler.fit_transform(X.reshape(-1, n_channels))
    else:
        X_2d = scaler.transform(X.reshape(-1, n_channels))

    X_t = torch.tensor(
        X_2d.reshape(X.shape), dtype=torch.float32
    ).permute(0, 2, 1)
    y_t = torch.tensor(y, dtype=torch.long)
    return X_t, y_t


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            loss = criterion(model(xb), yb)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss


def evaluate(model, X_t, y_t, criterion, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_t.to(device))
        loss = criterion(logits, y_t.to(device)).item()
        preds = logits.argmax(dim=1).cpu()
        acc = (preds == y_t).float().mean().item()
    return loss, acc, preds.numpy()


if __name__ == "__main__":
    data_dir = "./dataset_rats"

    path = os.path.join(data_dir, f"rat{RAT_ID}.npz")
    rms, angles, _ = load_rat_file(path)
    X, y = prepare_dataset(rms, angles)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data_temporal(X, y)
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    n_channels = X_train.shape[-1]
    scaler = StandardScaler()

    X_train_t, y_train_t = to_tensors(X_train, y_train, scaler, n_channels, fit_scaler=True)
    X_val_t,   y_val_t   = to_tensors(X_val,   y_val,   scaler, n_channels)
    X_test_t,  y_test_t  = to_tensors(X_test,  y_test,  scaler, n_channels)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True
    )

    device = torch.device("cpu")
    model = CNN1D(n_channels=n_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = evaluate(model, X_val_t, y_val_t, criterion, device)
        print(f"  Epoch {epoch+1}/{EPOCHS} — "
              f"train_loss: {train_loss/len(y_train):.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nRestoring best checkpoint (val_loss: {best_val_loss:.4f})")
    model.load_state_dict(best_state)

    test_loss, test_acc, test_preds = evaluate(model, X_test_t, y_test_t, criterion, device)
    print(f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
    print(classification_report(y_test, test_preds, target_names=["Neutral", "Dorsi", "Plantar"]))
