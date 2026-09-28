"""Build the pretraining mix: Imagenette + Galaxy10 DECaLS + EuroSAT, all 96x96 RGB.

    python training_data.py            # -> data/train (datasets.save_to_disk)

Load it with ``datasets.load_from_disk("data/train")`` or ``spt.data.HFDataset("data/train")``.
"""

import numpy as np
from datasets import Features, Image, Value, concatenate_datasets, load_dataset

SIZE = 96
PROBE_PER_CLASS = 100  # per class, held out from training for the evaluation probe (see evaluation_data.py)

_INET10 = "https://huggingface.co/datasets/frgfm/imagenette/resolve/4b23ffb92a8029db9958bdfdbd6978427d09b1a0/160px"
# name -> (load_dataset args, load_dataset kwargs, center-crop fraction of the short side)
SOURCES = {
    "inet10": (("parquet",), {"data_files": {"train": f"{_INET10}/train-00000-of-00001.parquet",
                                             "test": f"{_INET10}/validation-00000-of-00001.parquet"}}, 1.0),
    "galaxy10": (("matthieulel/galaxy10_decals",), {}, 0.7),
    "eurosat": (("tanganke/eurosat",), {}, 1.0),
}
DOMAINS = list(SOURCES)
FEATURES = Features({"image": Image(), "label": Value("int64"), "domain": Value("int64"),
                     "global_label": Value("int64")})


def preprocess(img, crop=1.0, size=SIZE):
    """Center-crop `crop` of the short side, resize to `size` x `size` RGB."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h) * crop
    box = ((w - s) / 2, (h - s) / 2, (w + s) / 2, (h + s) / 2)
    return img.resize((size, size), resample=3, box=box)  # 3 = bicubic


def load_source(name, split):
    args, kwargs, crop = SOURCES[name]
    return load_dataset(*args, split=split, **kwargs), crop


def split_train(labels, seed=0, per_domain=10_000):
    """Seeded split of a train split into (pretraining, probe) indices. Shared with evaluation_data.py."""
    order = np.random.default_rng(seed).permutation(len(labels))
    labels = np.asarray(labels)[order]
    rank, seen = np.empty(len(labels), dtype=int), {}
    for i, l in enumerate(labels):  # rank = position among samples of the same class
        rank[i] = seen[l] = seen.get(l, -1) + 1
    probe = order[rank < PROBE_PER_CLASS]
    train = order[rank >= PROBE_PER_CLASS][:per_domain]
    return np.sort(train), np.sort(probe)


def make_training_data(out_dir="data/train", per_domain=10_000, seed=0, num_proc=4):
    parts = []
    for d, name in enumerate(DOMAINS):
        ds, crop = load_source(name, "train")
        train_idx, _ = split_train(ds["label"], seed, per_domain)
        ds = ds.select(train_idx).map(
            lambda b: {"image": [preprocess(im, crop) for im in b["image"]],
                       "label": b["label"], "domain": [d] * len(b["label"]),
                       "global_label": [10 * d + l for l in b["label"]]},
            batched=True, remove_columns=ds.column_names, features=FEATURES, num_proc=num_proc,
        )
        parts.append(ds)
    ds = concatenate_datasets(parts).shuffle(seed=seed).flatten_indices()
    ds.save_to_disk(out_dir)
    return ds


if __name__ == "__main__":
    ds = make_training_data()
    print(ds, np.bincount(ds["domain"]))
