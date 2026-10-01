import json
import os

import numpy as np
import onnxruntime as ort
import pytest
import torch
from conftest import DOMAINS, MODEL, SAMPLE, last_json, run
from datasets import load_from_disk

from evaluate import DIM, Ridge, SubmissionError, embed


def test_ridge_separates_classes():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 25)
    x = np.eye(4)[y] * 5 + rng.normal(size=(len(y), 4))
    x = np.concatenate([x, rng.normal(size=(len(y), 60))], 1)  # plus noise features
    assert (Ridge().fit(x, y).predict(x) == y).mean() > 0.95


def test_ridge_keeps_original_labels():
    rng = np.random.default_rng(0)
    y = np.repeat([3, 7, 11], 10)
    x = rng.normal(size=(len(y), 8)) + y[:, None]
    assert set(Ridge().fit(x, y).predict(x)) <= {3, 7, 11}


def test_embed_sample_data():
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    z = embed(sess, load_from_disk(str(SAMPLE)), batch=2)  # batch < len exercises the batching
    assert z.shape == (3, DIM) and np.isfinite(z).all()


def test_example_submission_on_synthetic_data(eval_dir, tmp_path):
    out = tmp_path / "results.json"
    r = run("evaluate.py", MODEL, "--data", eval_dir, "--out", out, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    score = last_json(r.stdout)["score"]

    res = json.loads(out.read_text())
    assert set(res) == {"summary", "id", "novel"}  # shift splits are ignored
    assert set(res["id"]) == set(DOMAINS) and set(res["novel"]) == {"toy"}
    s = res["summary"]
    assert set(s) == {"id", "novel", "score", "seconds"}
    assert score == s["score"] == pytest.approx((s["id"] + s["novel"]) / 2)
    assert score > 0.6  # 3-way classes separated by colour; chance is 1/3


class Flat(torch.nn.Module):
    def __init__(self, dim=DIM, nan=False):
        super().__init__()
        self.lin, self.nan = torch.nn.Linear(3, dim), nan

    def forward(self, x):
        z = self.lin(x.mean((2, 3)))
        return z * float("nan") if self.nan else z


def export(model, path, input_name="image"):
    torch.onnx.export(model.eval(), (torch.randn(2, 3, 96, 96),), path, input_names=[input_name],
                      output_names=["embedding"], dynamic_axes={input_name: {0: "batch"}}, opset_version=17,
                      dynamo=False)
    return path


@pytest.mark.parametrize("model, message", [
    (Flat(dim=256), "expected (batch, 1024)"),
    (Flat(nan=True), "NaN or infinite"),
])
def test_embed_rejects_bad_outputs(tmp_path, model, message):
    sess = ort.InferenceSession(str(export(model, tmp_path / "bad.onnx")), providers=["CPUExecutionProvider"])
    with pytest.raises(SubmissionError, match=message.replace("(", r"\(").replace(")", r"\)")):
        embed(sess, load_from_disk(str(SAMPLE)))


@pytest.mark.parametrize("model, input_name, message", [
    (Flat(), "pixels", "exactly one input, named 'image'"),
    (Flat(dim=256), "image", "expected (batch, 1024)"),
    (Flat(nan=True), "image", "NaN or infinite"),
])
def test_bad_submissions_report_error(eval_dir, tmp_path, model, input_name, message):
    path = export(model, tmp_path / "bad.onnx", input_name)
    r = run("evaluate.py", path, "--data", eval_dir, cwd=tmp_path)
    assert r.returncode == 1
    assert message in last_json(r.stdout)["error"]


def test_unloadable_model_reports_error(eval_dir, tmp_path):
    path = tmp_path / "junk.onnx"
    path.write_bytes(os.urandom(64))
    r = run("evaluate.py", path, "--data", eval_dir, cwd=tmp_path)
    assert r.returncode == 1
    assert "Could not load the ONNX model" in last_json(r.stdout)["error"]


def test_missing_data_dir(tmp_path):
    r = run("evaluate.py", MODEL, "--data", tmp_path / "nope", cwd=tmp_path)
    assert r.returncode == 1 and "pass --data" in r.stderr
