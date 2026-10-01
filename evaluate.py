"""CPU evaluation of an ONNX encoder with a closed-form linear (ridge) probe.

    python evaluate.py model.onnx [--data data/eval] [--out results.json]

The script is self-contained so it can be uploaded as a challenge evaluator. The runner starts it from a temp
directory, so pass the data location explicitly there:

    stable-challenge evaluate NAME -- --data /abs/path/to/data/eval

Problems with the submitted model are reported as {"error": ...} on stdout, which the leaderboard shows.

Prints {"score": <float>} as JSON on stdout; the detailed report goes to stderr.

Score = mean of
    id     probe fit on probe_train, accuracy on id_test                  (per domain, averaged)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
from datasets import load_from_disk

SIZE, DIM = 96, 1024
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

LAMBDAS = np.logspace(-4, 2, 13)  # relative to the mean eigenvalue of the Gram matrix
SHOTS, EPISODES = 10, 5


class SubmissionError(Exception):
    """A problem with the submitted model, shown to the participant."""


def embed(session, ds, batch=256):
    x = np.stack([np.asarray(im, dtype=np.float32) for im in ds["image"]]) / 255.0
    x = ((x - MEAN) / STD).astype(np.float32).transpose(0, 3, 1, 2)
    out = []
    for i in range(0, len(x), batch):
        try:
            z = session.run(None, {"image": x[i : i + batch]})[0]
        except Exception as e:
            raise SubmissionError(
                f"Model failed on float32 input of shape {x[i : i + batch].shape}: {e}"
            ) from None
        if z.shape != (len(x[i : i + batch]), DIM):
            raise SubmissionError(f"Embedding shape {z.shape}, expected (batch, {DIM})")
        if not np.isfinite(z).all():
            raise SubmissionError("Embeddings contain NaN or infinite values")
        out.append(z)
    return np.concatenate(out)


class Ridge:
    """Closed-form ridge regression onto one-hot labels; argmax gives the class.

    Solved in the dual (n x n Gram matrix, one eigendecomposition). Lambda is picked by the exact
    leave-one-out error, which is also closed form, so there is no iterative optimisation anywhere.
    """

    def fit(self, x, y):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        x = (x - self.mu) / self.sd
        self.classes = np.unique(y)
        Y = (y[:, None] == self.classes).astype(np.float64)
        self.ybar = Y.mean(0)
        Yc = Y - self.ybar
        e, U = np.linalg.eigh(x @ x.T)
        e = np.clip(e, 0, None)
        UtY = U.T @ Yc
        best = None
        for lam in LAMBDAS * e.mean():
            h = e / (e + lam)
            H_diag = (U**2) @ h + 1 / len(x)  # leverage, incl. the intercept
            loo = (Yc - U @ (h[:, None] * UtY)) / (1 - H_diag)[:, None]
            err = (loo**2).sum()
            if best is None or err < best[0]:
                best = (err, lam)
        self.lam = best[1]
        self.W = x.T @ (U @ (UtY / (e + self.lam)[:, None]))
        return self

    def predict(self, x):
        return self.classes[(((x - self.mu) / self.sd) @ self.W + self.ybar).argmax(1)]


def evaluate(model_path, data="data/eval", seed=0):
    t0 = time.time()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count()
    try:
        sess = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        raise SubmissionError(f"Could not load the ONNX model: {e}") from None
    if [i.name for i in sess.get_inputs()] != ["image"]:
        raise SubmissionError("ONNX model must have exactly one input, named 'image'")
    splits = {
        name: load_from_disk(os.path.join(data, name))
        for name in sorted(os.listdir(data))
        if not name.startswith("shift_")
    }
    feats = {name: embed(sess, ds) for name, ds in splits.items()}
    t_embed = time.time() - t0

    labels = {name: np.asarray(ds["label"]) for name, ds in splits.items()}
    domain = {name: np.asarray(ds["domain"]) for name, ds in splits.items()}
    res = defaultdict(dict)
    for d in np.unique(domain["probe_train"]):
        m = domain["probe_train"] == d
        probe = Ridge().fit(feats["probe_train"][m], labels["probe_train"][m])
        t = domain["id_test"] == d
        res["id"][d] = float(
            (probe.predict(feats["id_test"][t]) == labels["id_test"][t]).mean()
        )

    rng = np.random.default_rng(seed)
    for name in splits:
        if not name.startswith("novel_"):
            continue
        role, y, z = np.asarray(splits[name]["role"]), labels[name], feats[name]
        sup, qry = np.flatnonzero(role == "support"), np.flatnonzero(role == "query")
        accs = []
        for _ in range(EPISODES):
            idx = np.concatenate(
                [rng.permutation(sup[y[sup] == c])[:SHOTS] for c in np.unique(y)]
            )
            accs.append((Ridge().fit(z[idx], y[idx]).predict(z[qry]) == y[qry]).mean())
        res["novel"][name[6:]] = float(np.mean(accs))

    summary = {k: float(np.mean(list(res[k].values()))) for k in ("id", "novel")}
    summary["score"] = float(np.mean(list(summary.values())))
    summary["seconds"] = {
        "embed": round(t_embed, 1),
        "total": round(time.time() - t0, 1),
    }
    return {"summary": summary, **res}


def print_report(r, file=sys.stderr):
    lines = [f"{'metric':28s} {'acc':>6s}"]
    for group in ("id", "novel"):
        for k, v in r[group].items():
            lines.append(f"  {group + '/' + k:26s} {100 * v:6.1f}")
    s = r["summary"]
    lines.append("-" * 36)
    for k in ("id", "novel", "score"):
        lines.append(f"{k:28s} {100 * s[k]:6.1f}")
    lines.append(f"time: {s['seconds']}")
    print("\n".join(lines), file=file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to the ONNX model")
    ap.add_argument(
        "--data", default="data/eval", help="directory of evaluation splits"
    )
    ap.add_argument("--out", default=None, help="also write the full results JSON here")
    a = ap.parse_args()
    if not os.path.isdir(a.data):
        sys.exit(f"Evaluation data not found at {a.data!r}; pass --data")
    try:
        r = evaluate(a.model, a.data)
    except SubmissionError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    print_report(r)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
    print(json.dumps({"score": r["summary"]["score"]}))
