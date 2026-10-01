## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv sync` creates
`.venv` from [pyproject.toml](pyproject.toml), with the exact versions pinned in `uv.lock`.

```bash
uv sync                                                    # create .venv with all dependencies
uv run training_data.py                                    # -> data/train   (28,469 images, 96x96)
uv run test_submission.py example_submission/model.onnx    # check a model before submitting it
```

Use `uv run` in front of any command (e.g. `uv run python my_training.py`), or activate the environment with
`source .venv/bin/activate`. Add packages you need for training with `uv add <package>`.

## Training data

`make_training_data()` in [training_data.py](training_data.py) builds one Hugging Face dataset with the columns
`image` (96×96 RGB), `label` (0–9 within its domain), `domain` (0 = Imagenette, 1 = Galaxy10, 2 = EuroSAT) and
`global_label` (0–29). Up to 10,000 images come from each domain. 100 images per class are held out from every
train split for the evaluation probe.

The data is meant for self-supervised learning. You may use the labels for monitoring, for example with
stable-pretraining's `OnlineProbe`, but not in the training objective. Load it with
`datasets.load_from_disk("data/train")` or `spt.data.HFDataset("data/train")`.


## Submission 

Submissions should include an ONNX format model with input and output dimensions below. Before submitting, make sure `uv run test_submission.py model.onnx` passes (see below).

## Model Format
An ONNX file with:

- **input** `image`: float32, `[B, 3, 96, 96]`, RGB scaled to [0, 1] then normalised with the ImageNet mean/std
- **output** `embedding`: float32, `[B, 1024]`

For example: `export_onnx(encoder, path)` in [export_onnx.py](export_onnx.py) exports a PyTorch encoder. If the encoder outputs
fewer than 1024 features it zero-pads them, which doesn't change the probe. It also checks the file with
onnxruntime.


## Validate submission

Check your model before uploading it:

```bash
uv run test_submission.py path/to/model.onnx
```

It runs the same steps as the official evaluation and explains how to fix any problem it finds. The checks are
listed at the top of [test_submission.py](test_submission.py). Only submit once validation passes.
