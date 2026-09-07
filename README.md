# DeepFake Face detector

A PyTorch image classification project that trains models to distinguish
real photographs from AI-generated ("deepfake") faces, using the
[DeepFakeFusion 399k Real/Fake Faces](https://www.kaggle.com/datasets/ajaysonicu/deepfakefusion-399k-realfake-faces)
dataset from Kaggle.

**[Try Live Demo](https://deepfake-face-authenticator.onrender.com/)**

Three model options are supported:
- **cnn** — a small custom CNN trained from scratch (`TinyVGG`)
- **efficientnet** — pretrained EfficientNet-B0, fine-tuned (transfer learning, default)

## Project structure

```
.
├── app/                # FastAPI application for inference
│   ├── main.py
│   ├── config.py
│   ├── inference.py
│   └── schemas.py
├── src/                # Training source code
│   ├── download_data.py
│   ├── data_setup.py
│   ├── model_builder.py
│   ├── engine.py
│   ├── utils.py
│   └── train.py
├── notebooks/          # Jupyter notebooks
│   └── Coded_by_me.ipynb
├── models/             # Saved model weights (.pth) land here
├── results/            # Saved PNGs (loss/accuracy curves, prediction grids) land here
├── Dockerfile          # Containerization for the app
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### Kaggle credentials

Get your API key from Kaggle: **Account → API → Create New Token**.
Then set these as environment variables (never hardcode them in code or notebooks):

```python
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_key'
```

In Colab, prefer using **Secrets** (key icon in the sidebar) and pulling values with
`google.colab.userdata.get(...)` instead of typing your key into a cell.

## Usage

```bash
cd src
python train.py --model efficientnet --epochs 5 --batch-size 32
python train.py --model cnn --epochs 10 --image-size 128
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/deepfakefusion_v2_sixsource_dataset` | Root folder with `train/`, `val/`, `test/` subfolders (ImageFolder format: `class_name/*.jpg`) |
| `--model` | `efficientnet` | One of `cnn`, `efficientnet` |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--image-size` | `128` | Images are resized to `image-size x image-size` |
| `--models-dir` | `models` | Where trained weights are saved |
| `--results-dir` | `results` | Where PNG plots are saved |

## Outputs

After training, you'll find:
- `models/<model>_model.pth` — trained model weights (`state_dict`)
- `results/<model>_curves.png` — loss + accuracy curves over training
- `results/<model>_predictions.png` — a labeled grid of sample predictions (green = correct, red = incorrect)

## Notes

- Transfer learning (`efficientnet`) converges much faster and
  with less data than training the CNN from scratch — start there.
- The dataset's `fake` images come from several generators; class labels are
  collapsed to a binary `real` / `fake` target for training.
- A GPU (e.g. Colab's T4) is strongly recommended — CPU training on this
  dataset size will be very slow.
