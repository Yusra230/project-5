"""Training engine for the live breast-cancer training dashboard."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data(scale=True):
    ds = load_breast_cancer()
    # sklearn target: 0=malignant, 1=benign. Convert to 1=malignant.
    X = ds.data.astype(np.float32)
    y = (ds.target == 0).astype(np.int64)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    scaler = StandardScaler()
    if scale:
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
    return (torch.tensor(X_train), torch.tensor(y_train),
            torch.tensor(X_val), torch.tensor(y_val),
            torch.tensor(X_test), torch.tensor(y_test), scaler)

class CancerNet(nn.Module):
    def __init__(self, hidden=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

def diagnose(hist):
    if not hist["train"]:
        return "No training run yet"
    if hist["train"][-1] > 0.3:
        return "Underfitting"
    best = min(hist["val"])
    if hist["val"][-1] > best * 1.30:
        return "Overfitting"
    return "Healthy"

def train(cfg, data=None, on_epoch=None):
    """Train one experiment; callback receives (zero-based epoch, history)."""
    torch.manual_seed(int(cfg.get("seed", SEED)))
    if data is None:
        data = load_data(scale=cfg.get("scale", True))
    Xtr, ytr, Xva, yva, *_ = data
    model = CancerNet(int(cfg["hidden"]), float(cfg["dropout"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"])
    )
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train": [], "val": []}
    best_loss, best_state, wait, best_epoch = float("inf"), None, 0, 0
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(Xtr), ytr)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            tr = loss_fn(model(Xtr), ytr).item()
            va = loss_fn(model(Xva), yva).item()
        hist["train"].append(tr)
        hist["val"].append(va)
        if va < best_loss:
            best_loss, best_epoch = va, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if on_epoch:
            on_epoch(epoch + 1, hist)
        if patience > 0 and wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, {"epochs_run": len(hist["train"]),
                         "best_val_loss": float(best_loss),
                         "best_epoch": int(best_epoch),
                         "diagnosis": diagnose(hist)}

def evaluate(model, X, y, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)[:, 1].numpy()
    pred = (probs >= threshold).astype(int)
    y_np = y.numpy()
    tp = int(((pred == 1) & (y_np == 1)).sum())
    fp = int(((pred == 1) & (y_np == 0)).sum())
    fn = int(((pred == 0) & (y_np == 1)).sum())
    tn = int(((pred == 0) & (y_np == 0)).sum())
    return {"accuracy": float((pred == y_np).mean()),
            "cancers_found": f"{tp}/{tp+fn}",
            "false_alarms": fp, "true_negative": tn,
            "false_negative": fn, "probabilities": probs.tolist(),
            "predictions": pred.tolist()}
