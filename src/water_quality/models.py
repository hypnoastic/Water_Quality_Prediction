"""Small regularized MLP with explicit chronological early stopping."""

import copy
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, input_dim=16, widths=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        size = input_dim
        for width in widths:
            layers.extend([nn.Linear(size, width), nn.ReLU(), nn.Dropout(dropout)])
            size = width
        layers.append(nn.Linear(size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def predict_mlp(model, x, batch_size=16384):
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(np.asarray(x[start : start + batch_size]), dtype=torch.float32)
            out.append(model(batch).numpy() * 100)
    return np.clip(np.concatenate(out), 0, 100)


def fit_mlp(x, y, xv, yv, widths, config):
    seed_everything(config["seed"], config["threads"])
    model = MLP(x.shape[1], widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    tx = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    ty = torch.as_tensor(np.asarray(y) / 100, dtype=torch.float32)
    best_score, best_state, waiting = float("inf"), None, 0
    history = []
    for epoch in range(config["mlp_epochs"]):
        model.train()
        order = torch.randperm(len(tx))
        total = 0.0
        for ids in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(tx[ids]), ty[ids])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        prediction = predict_mlp(model, xv)
        score = float(np.sqrt(np.mean((prediction - np.asarray(yv)) ** 2)))
        history.append(
            {
                "epoch": epoch + 1,
                "train_rmse": np.sqrt(total / len(tx)) * 100,
                "validation_rmse": score,
            }
        )
        if score < best_score - 0.0001:
            best_score, best_state, waiting = score, copy.deepcopy(model.state_dict()), 0
        else:
            waiting += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  MLP {widths}, epoch {epoch + 1}: validation RMSE {score:.4f}", flush=True)
        if waiting >= config["mlp_patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history
