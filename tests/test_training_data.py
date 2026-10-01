import numpy as np
import pytest
from conftest import SAMPLE, images
from datasets import Dataset, Features, Image, Value, load_from_disk
from PIL import Image as PILImage

import training_data as td


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
@pytest.mark.parametrize("size", [(160, 120), (60, 200)])
def test_preprocess_makes_square_rgb(mode, size):
    out = td.preprocess(PILImage.new(mode, size), crop=0.7)
    assert out.mode == "RGB" and out.size == (td.SIZE, td.SIZE)


def test_preprocess_center_crops():
    img = np.zeros((100, 100, 3), np.uint8)
    img[25:75, 25:75] = 255  # white centre, black border
    cropped = np.asarray(td.preprocess(PILImage.fromarray(img), crop=0.5))
    full = np.asarray(td.preprocess(PILImage.fromarray(img), crop=1.0))
    assert cropped[4:-4, 4:-4].min() > 200  # only the white centre (bicubic blurs the outermost pixels)
    assert full[:10, :10].max() < 50  # without cropping the black border remains


def test_split_train_holds_out_probe_per_class():
    labels = np.repeat(np.arange(4), [150, 120, 101, 300])
    train, probe = td.split_train(labels, seed=0, per_domain=10_000)
    assert not set(train) & set(probe) and len(train) + len(probe) == len(labels)
    assert (np.bincount(labels[probe]) == td.PROBE_PER_CLASS).all()
    assert (np.diff(train) > 0).all() and (np.diff(probe) > 0).all()  # sorted, unique


def test_split_train_caps_and_is_seeded():
    labels = np.repeat(np.arange(3), 200)
    train, probe = td.split_train(labels, seed=0, per_domain=50)
    assert len(train) == 50
    again = td.split_train(labels, seed=0, per_domain=50)
    other = td.split_train(labels, seed=1, per_domain=50)
    assert (train == again[0]).all() and (probe == again[1]).all()
    assert not (probe == other[1]).all()


def fake_source(n_per_class, classes=10, size=40):
    labels = np.repeat(np.arange(classes), n_per_class)
    return Dataset.from_dict({"image": images(labels, np.random.default_rng(0), size=size),
                              "label": labels.tolist()},
                             features=Features({"image": Image(), "label": Value("int64")}))


def test_make_training_data(monkeypatch, tmp_path):
    monkeypatch.setattr(td, "PROBE_PER_CLASS", 2)
    sources = {"inet10": fake_source(5), "galaxy10": fake_source(4), "eurosat": fake_source(6)}
    monkeypatch.setattr(td, "load_source", lambda name, split: (sources[name], td.SOURCES[name][2]))

    ds = td.make_training_data(str(tmp_path), per_domain=25, num_proc=1)
    assert ds.features == td.FEATURES
    assert np.bincount(ds["domain"]).tolist() == [25, 20, 25]  # inet10/eurosat capped, galaxy10 has 10 * (4 - 2)
    assert (np.asarray(ds["global_label"]) == 10 * np.asarray(ds["domain"]) + np.asarray(ds["label"])).all()
    assert {im.size for im in ds["image"]} == {(td.SIZE, td.SIZE)}
    assert ds["domain"] != sorted(ds["domain"])  # shuffled across domains
    assert load_from_disk(str(tmp_path))["global_label"] == ds["global_label"]


def test_sample_matches_training_format():
    ds = load_from_disk(str(SAMPLE))
    assert ds.features == td.FEATURES
    assert ds["domain"] == list(range(len(td.DOMAINS)))  # one image per domain
    assert {(im.mode, im.size) for im in ds["image"]} == {("RGB", (td.SIZE, td.SIZE))}
