"""
pachhaanna_torch.py  —  the "real data" version (Permuted MNIST)
================================================================

Same idea as pachhaanna.py (the recognition / *pachhaanna* approach to continual
learning) but on real MNIST with PyTorch, so you can see the effect at scale.

   NAIVE   : train each task on top of the last  -> catastrophic forgetting
   EWC     : recognise & protect the weights that carry old tasks (diagonal Fisher)
   REPLAY  : recall the past by interleaving a small buffer of remembered examples
   BOTH    : EWC + replay

>>> IMPORTANT: this file was NOT executed in the environment it was written in
    (no PyTorch, no GPU, no internet there). It is a standard implementation
    meant to be run locally:

        pip install torch torchvision
        python pachhaanna_torch.py

    The from-scratch NumPy version (pachhaanna.py) IS the one that was actually
    run, with the numbers and figure you were shown.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TASKS = 5
EPOCHS = 1
BATCH = 128
LR = 0.1
EWC_LAMBDA = 1000.0      # tune: higher = remembers more but learns new tasks less
REPLAY_BUFFER = 2000     # examples kept from the past
REPLAY_BATCH = 128


# ----------------------------- data: permuted MNIST -----------------------------
def load_mnist():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST(".", train=True, download=True, transform=tf)
    te = datasets.MNIST(".", train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    yte = torch.tensor([te[i][1] for i in range(len(te))])
    return Xtr, ytr, Xte, yte


def permuted_tasks(n_tasks, seed=0):
    Xtr, ytr, Xte, yte = load_mnist()
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for _ in range(n_tasks):
        p = torch.randperm(784, generator=g)
        tasks.append(dict(Xtr=Xtr[:, p], ytr=ytr, Xte=Xte[:, p], yte=yte))
    return tasks


# ----------------------------- model -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


# ----------------------------- EWC: recognise what mattered -----------------------------
class EWC:
    """Stores, per consolidated task, a snapshot of the weights and the diagonal
    Fisher information = which weights the model relied on for that task."""
    def __init__(self):
        self.tasks = []  # list of (params_snapshot, fisher)

    def consolidate(self, model, X, y, n=600):
        model.eval()
        fisher = {n_: torch.zeros_like(p) for n_, p in model.named_parameters()}
        idx = torch.randperm(len(X))[:n]
        for i in idx:
            model.zero_grad()
            out = model(X[i:i + 1].to(DEVICE))
            loss = F.cross_entropy(out, y[i:i + 1].to(DEVICE))
            loss.backward()
            for n_, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n_] += p.grad.detach() ** 2 / len(idx)
        snap = {n_: p.detach().clone() for n_, p in model.named_parameters()}
        self.tasks.append((snap, fisher))

    def penalty(self, model):
        loss = 0.0
        for snap, fisher in self.tasks:
            for n_, p in model.named_parameters():
                loss = loss + (fisher[n_] * (p - snap[n_]) ** 2).sum()
        return loss


# ----------------------------- replay buffer: recall the past -----------------------------
class ReplayBuffer:
    def __init__(self, cap): self.cap = cap; self.X = None; self.y = None

    def add(self, X, y):
        idx = torch.randperm(len(X))[: self.cap // N_TASKS]
        X, y = X[idx], y[idx]
        self.X = X if self.X is None else torch.cat([self.X, X])[-self.cap:]
        self.y = y if self.y is None else torch.cat([self.y, y])[-self.cap:]

    def sample(self, n):
        if self.X is None: return None, None
        idx = torch.randperm(len(self.X))[:n]
        return self.X[idx], self.y[idx]


# ----------------------------- train / eval -----------------------------
def evaluate(model, tasks):
    model.eval()
    accs = []
    with torch.no_grad():
        for t in tasks:
            pred = model(t["Xte"].to(DEVICE)).argmax(1).cpu()
            accs.append((pred == t["yte"]).float().mean().item())
    return accs


def run(strategy, tasks):
    model = MLP().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    ewc = EWC() if strategy in ("ewc", "both") else None
    buf = ReplayBuffer(REPLAY_BUFFER) if strategy in ("replay", "both") else None
    R = np.zeros((len(tasks), len(tasks)))

    for ti, task in enumerate(tasks):
        X, y = task["Xtr"], task["ytr"]
        model.train()
        for _ in range(EPOCHS):
            for s in range(0, len(X), BATCH):
                xb, yb = X[s:s + BATCH].to(DEVICE), y[s:s + BATCH].to(DEVICE)
                if buf is not None:
                    xr, yr = buf.sample(REPLAY_BATCH)
                    if xr is not None:
                        xb = torch.cat([xb, xr.to(DEVICE)]); yb = torch.cat([yb, yr.to(DEVICE)])
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                if ewc is not None and ewc.tasks:
                    loss = loss + EWC_LAMBDA * ewc.penalty(model)
                loss.backward()
                opt.step()
        if ewc is not None: ewc.consolidate(model, X, y)
        if buf is not None: buf.add(X, y)
        for tj in range(ti + 1):
            R[ti, tj] = evaluate(model, [tasks[tj]])[0]

    final = R[-1]
    forget = np.mean([R[j:, j].max() - R[-1, j] for j in range(len(tasks) - 1)])
    print(f"{strategy:<8} avg final acc={final.mean():.3f}  forgetting={forget:.3f}  "
          f"per-task={np.round(final,2)}")
    return R


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    tasks = permuted_tasks(N_TASKS, seed=0)
    for strat in ["naive", "ewc", "replay", "both"]:
        run(strat, tasks)
