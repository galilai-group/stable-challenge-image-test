"""Tiny synthetic datasets in the same on-disk format as the real training and evaluation data.

Each class has its own mean colour plus noise, so a reasonable encoder plus the ridge probe does well above chance.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, Features, Image, Value, concatenate_datasets

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "example_submission" / "model.onnx"
SAMPLE = ROOT / "data" / "sample"
DOMAINS, CLASSES, SIZE = ("inet10", "galaxy10", "eurosat"), 3, 96


def images(labels, rng, offset=0, size=SIZE):
    colours = np.random.default_rng(offset).uniform(40, 215, size=(max(labels) + 1, 3))
    x = colours[labels][:, None, None, :] + rng.normal(0, 25, size=(len(labels), size, size, 3))
    return list(np.clip(x, 0, 255).astype(np.uint8))


def split(per_class, rng, domain, offset, **extra):
    labels = np.repeat(np.arange(CLASSES), per_class)
    cols = {"image": images(labels, rng, offset), "label": labels.tolist(), "domain": [domain] * len(labels)}
    feats = {"image": Image(), "label": Value("int64"), "domain": Value("string")}
    for k, v in extra.items():
        cols[k], feats[k] = v, Value("string")
    return Dataset.from_dict(cols, features=Features(feats))


@pytest.fixture(scope="session")
def eval_dir(tmp_path_factory):
    out, rng = tmp_path_factory.mktemp("eval"), np.random.default_rng(0)
    concatenate_datasets([split(12, rng, d, i) for i, d in enumerate(DOMAINS)]).save_to_disk(out / "probe_train")
    concatenate_datasets([split(6, rng, d, i) for i, d in enumerate(DOMAINS)]).save_to_disk(out / "id_test")
    split(4, rng, "inet10", 0).save_to_disk(out / "shift_inet10_noise")  # must be ignored
    role = (["support"] * 12 + ["query"] * 4) * CLASSES  # per class: 12 support, 4 query
    split(16, rng, "novel", 9, role=role).save_to_disk(out / "novel_toy")
    return out


def run(script, *args, cwd):
    """Run a repo script with this interpreter from `cwd` (the challenge runner uses a temp dir)."""
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=cwd,
                          capture_output=True, text=True, timeout=600)


def last_json(stdout):
    return json.loads(stdout.strip().splitlines()[-1])
