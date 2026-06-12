import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import optuna
import os

EPOCHS = 50
N_TRIALS = 10


class CNN1D(nn.Module):
    def __init__(self, n_channels, filters1, filters2, fc_units, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, filters1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(filters1, filters2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filters2, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, 3)
        )

    def forward(self, x):
        return self.net(x)


def prepare_dataset(rms_data, angles_ds, seq_len):
    rms_data = rms_data[:56, :]
    X = rms_data.T

    y = np.zeros(len(angles_ds), dtype=np.int64)
    y[angles_ds > 2.0] = 1
    y[angles_ds < -2.0] = 2

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
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


def train_model(X_train, y_train, X_val, y_val, params, verbose=False):
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
        batch_size=params["batch_size"],
        shuffle=True
    )

    device = torch.device("cpu")
    model = CNN1D(
        n_channels=n_channels,
        filters1=params["filters1"],
        filters2=params["filters2"],
        fc_units=params["fc_units"],
        kernel_size=params["kernel_size"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )
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

        if verbose:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t.to(device))
                val_loss = criterion(val_logits, y_val_t.to(device)).item()
                val_preds = val_logits.argmax(dim=1).cpu()
                val_acc = (val_preds == y_val_t).float().mean().item()
            avg_train_loss = train_loss / len(y_train)
            print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {avg_train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t.to(device)).argmax(dim=1).cpu().numpy()

    return model, scaler, y_val, val_preds


def run_optuna(X_train, y_train, X_val, y_val, seed=42):
    def objective(trial):
        params = {
            "seq_len": trial.suggest_int("seq_len", 8, 64, step=8),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "filters1": trial.suggest_categorical("filters1", [8, 16, 32, 64]),
            "filters2": trial.suggest_categorical("filters2", [16, 32, 64, 128]),
            "fc_units": trial.suggest_categorical("fc_units", [16, 32, 64]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        }

        _, _, y_true, y_pred = train_model(X_train, y_train, X_val, y_val, params)
        acc = (y_pred == y_true).mean()
        return acc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest trial val_acc: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def leave_one_rat_out(data_dir, rat_ids, seed=42):
    results = {}

    for test_rat in rat_ids:
        print(f"\n{'='*50}\nTesting on RAT {test_rat}\n{'='*50}")

        X_train_list, y_train_list = [], []
        X_test, y_test = None, None

        for rat in rat_ids:
            path = os.path.join(data_dir, f"rat{rat}.npz")
            rms, angles, split_idx = load_rat_file(path)
            X, y = prepare_dataset(rms, angles, seq_len=16)

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
        print("Running Optuna hyperparameter search...")

        best_params = run_optuna(X_train, y_train, X_val, y_val, seed=seed)

        X_all_train = np.concatenate([X_train, X_val])
        y_all_train = np.concatenate([y_train, y_val])
        X_train_f, y_train_f, X_val_f, y_val_f, _, _ = split_data(
            X_all_train, y_all_train, train=0.85, val=0.15, seed=seed
        )

        model, scaler, y_true, y_pred = train_model(
            X_train_f, y_train_f, X_val_f, y_val_f, best_params, verbose=True
        )

        print(classification_report(y_true, y_pred, target_names=["Neutral", "Dorsi", "Plantar"]))
        results[test_rat] = {
            "report": classification_report(y_true, y_pred, output_dict=True),
            "best_params": best_params
        }

    return results


if __name__ == "__main__":
    data_dir = "./dataset_rats"
    rat_ids = list(range(4, 11))

    results = leave_one_rat_out(data_dir, rat_ids)

    print("\n\nBest params per rat:")
    for rat, res in results.items():
        print(f"  RAT {rat}: {res['best_params']}")
