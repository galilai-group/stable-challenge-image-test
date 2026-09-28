# stable-challenge-image-test


## Quick start

```bash
pip install -r requirements.txt
python training_data.py                              # -> data/train   (28,469 images, 96x96)
python evaluation_data.py                            # -> data/eval/*  (12 fixed splits)
```

## Training data

`make_training_data()` in [training_data.py](training_data.py) builds one Hugging Face dataset with the columns
`image` (96×96 RGB), `label` (0–9 within its domain), `domain` (0 = Imagenette, 1 = Galaxy10, 2 = EuroSAT) and
`global_label` (0–29). Up to 10,000 images come from each domain. 100 images per class are held out from every
train split for the evaluation probe.

The data is meant for self-supervised learning. You may use the labels for monitoring, for example with
stable-pretraining's `OnlineProbe`, but not in the training objective. Load it with
`datasets.load_from_disk("data/train")` or `spt.data.HFDataset("data/train")`.

## Submission format

An ONNX file with:

- **input** `image`: float32, `[B, 3, 96, 96]`, RGB scaled to [0, 1] then normalised with the ImageNet mean/std
- **output** `embedding`: float32, `[B, 1024]`

For an example, see `export_onnx(encoder, path)` in [export_onnx.py](export_onnx.py) exports a PyTorch encoder. If the encoder outputs
fewer than 1024 features it zero-pads them, which doesn't change the probe. It also checks the file with
onnxruntime.

