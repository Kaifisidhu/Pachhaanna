"""
pachhaanna.py
=============
Continual learning without catastrophic forgetting, built around the idea of
*pachhaanna* (ਪਛਾਣ) -- recognition: integrating the new by RELATING it to what
is already known, instead of overwriting it.

Three mechanisms, each a literal reading of "recognition":

  1. NAIVE              -- no recognition. Each new task is trained straight on
                           top of the last. The shared layers get overwritten and
                           the model forgets earlier tasks. (The control.)

  2. EWC (consolidation)-- after a task, the model estimates WHICH of its own
                           weights carry what it just learned (the diagonal Fisher
                           information) and anchors them. While learning the next
                           task it pays a penalty for moving those weights. This is
                           "recognising what mattered and refusing to overwrite it."
                           (Kirkpatrick et al., PNAS 2017 -- implemented from scratch.)

  3. PROTOTYPE REPLAY   -- after a task, the model stores a compressed *essence* of
                           each class it saw (a Gaussian: mean + variance in input
                           space). While learning later tasks it RE-COGNISES the old
                           by regenerating samples from those essences and weaving
                           them back into training -- a small consolidation / "dreaming"
                           loop. (Feature/prototype replay.)

The "pachhaanna" learner = EWC + prototype replay together.

No internet, no downloads, no torch -- pure NumPy on a synthetic permuted benchmark
(the standard catastrophic-forgetting setup: one base problem seen through a
different fixed feature-permutation per task, shared output head).
"""

import numpy as np


# ----------------------------------------------------------------------------- #
#  Synthetic continual-learning benchmark (Permuted-MNIST analogue, no download)
# ----------------------------------------------------------------------------- #
def make_centers(d, k, sep, rng):
    return rng.normal(0.0, sep, size=(k, d))


def sample_blobs(centers, n_per_class, rng):
    """Sample unit-variance Gaussian blobs around the given class centers."""
    k, d = centers.shape
    X = np.concatenate([rng.normal(0, 1, (n_per_class, d)) + centers[c] for c in range(k)])
    y = np.concatenate([np.full(n_per_class, c) for c in range(k)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_permuted_tasks(T, d, k, n_train, n_test, sep, rng):
    """One base problem; task t applies a fixed random permutation of the d features.
    Train and test are drawn from the SAME class centers (aligned distributions)."""
    centers = make_centers(d, k, sep, rng)
    Xtr, ytr = sample_blobs(centers, n_train // k, rng)
    Xte, yte = sample_blobs(centers, n_test // k, rng)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8          # standardize (stable training)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    tasks = []
    for _ in range(T):
        perm = rng.permutation(d)
        tasks.append(dict(Xtr=Xtr[:, perm].copy(), ytr=ytr.copy(),
                          Xte=Xte[:, perm].copy(), yte=yte.copy()))
    return tasks


# ----------------------------------------------------------------------------- #
#  A small MLP with manual forward / backward (so every gradient is explicit)
# ----------------------------------------------------------------------------- #
class MLP:
    def __init__(self, sizes, rng):
        self.sizes = sizes
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            self.W.append(rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.L = len(self.W)

    # --- forward, caching inputs/pre-activations for backprop ---
    def forward(self, X):
        a, cache = X, []
        for i in range(self.L):
            z = a @ self.W[i] + self.b[i]
            cache.append((a, z))
            a = np.maximum(z, 0) if i < self.L - 1 else z   # ReLU hidden, linear logits
        return a, cache

    @staticmethod
    def softmax_ce(logits, y):
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        dlogits = p.copy(); dlogits[np.arange(n), y] -= 1; dlogits /= n
        return loss, dlogits, p

    # --- backward: returns grads matching W,b shapes ---
    def backward(self, cache, dlogits):
        dW = [None] * self.L; db = [None] * self.L
        dz = dlogits
        for i in reversed(range(self.L)):
            a_in, _ = cache[i]
            dW[i] = a_in.T @ dz
            db[i] = dz.sum(0)
            if i > 0:
                da = dz @ self.W[i].T
                _, z_prev = cache[i - 1]
                dz = da * (z_prev > 0)              # ReLU'
        return dW, db

    def predict(self, X):
        return self.forward(X)[0].argmax(1)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    def clone_params(self):
        return [w.copy() for w in self.W] + [b.copy() for b in self.b]


# ----------------------------------------------------------------------------- #
#  Diagonal empirical Fisher  ==  "which of my weights carry what I just learned"
# ----------------------------------------------------------------------------- #
def fisher_diagonal(net, X, y, n_samples, rng):
    FW = [np.zeros_like(w) for w in net.W]
    Fb = [np.zeros_like(b) for b in net.b]
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    for j in idx:                                   # per-example squared grads
        xj = X[j:j + 1]; yj = y[j:j + 1]
        logits, cache = net.forward(xj)
        _, dlogits, _ = net.softmax_ce(logits, yj)
        dW, db = net.backward(cache, dlogits)
        for l in range(net.L):
            FW[l] += dW[l] ** 2
            Fb[l] += db[l] ** 2
    m = len(idx)
    FW = [f / m for f in FW]
    Fb = [f / m for f in Fb]
    # normalize to a consistent scale so the EWC strength (lambda) is interpretable
    scale = (sum(f.sum() for f in FW) + sum(f.sum() for f in Fb)) / \
            (sum(f.size for f in FW) + sum(f.size for f in Fb)) + 1e-12
    return [f / scale for f in FW], [f / scale for f in Fb]


# ----------------------------------------------------------------------------- #
#  Prototype memory  ==  the compressed "essence" of each thing seen, for recall
# ----------------------------------------------------------------------------- #
class PrototypeMemory:
    """Stores, per (task,class), a Gaussian essence (mean,var) of the inputs."""
    def __init__(self): self.protos = []           # list of (mean, var, label)

    def consolidate(self, X, y):
        for c in np.unique(y):
            Xc = X[y == c]
            self.protos.append((Xc.mean(0), Xc.var(0) + 1e-3, int(c)))

    def replay_batch(self, n, rng):
        """Re-cognise the past: regenerate samples from stored essences."""
        if not self.protos:
            return None, None
        picks = rng.integers(0, len(self.protos), size=n)
        Xs = np.stack([self.protos[p][0] + rng.normal(0, 1, self.protos[p][0].shape)
                       * np.sqrt(self.protos[p][1]) for p in picks])
        ys = np.array([self.protos[p][2] for p in picks])
        return Xs, ys

    def recognise(self, X):
        """Nearest-essence classification -- pure recognition, no gradient."""
        means = np.stack([m for m, _, _ in self.protos])
        labs = np.array([c for _, _, c in self.protos])
        d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(2)
        return labs[d2.argmin(1)]


# ----------------------------------------------------------------------------- #
#  Training one task, with optional EWC penalty and optional prototype replay
# ----------------------------------------------------------------------------- #
def train_task(net, task, epochs, lr, batch, rng,
               ewc_store=None, ewc_lambda=0.0,
               proto_mem=None, replay_frac=0.0):
    X, y = task["Xtr"], task["ytr"]
    n = len(X)
    vW = [np.zeros_like(w) for w in net.W]          # momentum buffers
    vb = [np.zeros_like(b) for b in net.b]
    mom = 0.9
    for _ in range(epochs):
        for s in range(0, n, batch):
            xb, yb = X[s:s + batch], y[s:s + batch]

            # weave regenerated old samples into the batch (recognition / replay)
            if proto_mem is not None and replay_frac > 0:
                Xr, yr = proto_mem.replay_batch(int(len(xb) * replay_frac), rng)
                if Xr is not None:
                    xb = np.concatenate([xb, Xr]); yb = np.concatenate([yb, yr])

            logits, cache = net.forward(xb)
            _, dlogits, _ = net.softmax_ce(logits, yb)
            dW, db = net.backward(cache, dlogits)

            # EWC: penalise moving weights that carry old tasks
            if ewc_store and ewc_lambda > 0:
                for (sW, sb, FW, Fb) in ewc_store:
                    for l in range(net.L):
                        dW[l] += ewc_lambda * FW[l] * (net.W[l] - sW[l])
                        db[l] += ewc_lambda * Fb[l] * (net.b[l] - sb[l])

            # global-norm gradient clipping (keeps EWC + replay numerically stable)
            gn = np.sqrt(sum((g ** 2).sum() for g in dW) + sum((g ** 2).sum() for g in db))
            clip = 10.0
            if gn > clip:
                f = clip / (gn + 1e-12)
                dW = [g * f for g in dW]; db = [g * f for g in db]

            for l in range(net.L):                  # SGD + momentum
                vW[l] = mom * vW[l] - lr * dW[l]; net.W[l] += vW[l]
                vb[l] = mom * vb[l] - lr * db[l]; net.b[l] += vb[l]


# ----------------------------------------------------------------------------- #
#  Run a full continual sequence under one strategy; return accuracy matrix R
#  R[i, j] = test accuracy on task j AFTER finishing task i
# ----------------------------------------------------------------------------- #
def run_strategy(name, tasks, sizes, seed,
                 epochs=25, lr=0.05, batch=128,
                 use_ewc=False, ewc_lambda=0.0, fisher_n=400,
                 use_replay=False, replay_frac=0.5):
    rng = np.random.default_rng(seed)
    net = MLP(sizes, rng)
    ewc_store = [] if use_ewc else None
    mem = PrototypeMemory() if use_replay else None
    T = len(tasks)
    R = np.zeros((T, T))

    for i, task in enumerate(tasks):
        train_task(net, task, epochs, lr, batch, rng,
                   ewc_store=ewc_store, ewc_lambda=ewc_lambda,
                   proto_mem=mem, replay_frac=(replay_frac if use_replay else 0.0))
        if use_ewc:                                  # consolidate Fisher + anchor
            FW, Fb = fisher_diagonal(net, task["Xtr"], task["ytr"], fisher_n, rng)
            ewc_store.append((net.clone_params()[:net.L],
                              net.clone_params()[net.L:], FW, Fb))
        if use_replay:
            mem.consolidate(task["Xtr"], task["ytr"])
        for j in range(i + 1):
            R[i, j] = net.accuracy(tasks[j]["Xte"], tasks[j]["yte"])
    return R, net, mem


def summarise(name, R):
    T = R.shape[0]
    final = R[-1, :]                                 # acc on every task at the end
    avg = final.mean()
    # forgetting_j = best-ever acc on task j  -  final acc on task j
    forget = np.mean([R[j:, j].max() - R[-1, j] for j in range(T - 1)])
    print(f"{name:<26}  avg final acc = {avg:6.3f}   mean forgetting = {forget:6.3f}")
    print("   final per-task acc: " + "  ".join(f"T{j+1}:{final[j]:.2f}" for j in range(T)))
    return avg, forget, final
