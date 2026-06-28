# Anime Face CNN Classifier

A simple Convolutional Neural Network (CNN), built with PyTorch, that classifies
anime character face images. The example dataset is a two-class subset
(**Anastasia** vs. **Takao**) drawn from the dataset described in
[*AniWho: A Quick and Accurate Way to Classify Anime Character Faces in Images*](https://arxiv.org/pdf/2208.11012v3),
sourced from Danbooru.

## Project structure

```
anime-cnn-classifier/
├── src/
│   ├── dataset.py     # Data download/loading + custom AnimeDataset class
│   ├── model.py        # AnimeCNN architecture (nn.Module)
│   ├── train.py         # Training loop, dataloader split, loss plotting
│   ├── evaluate.py     # Prediction visualization utilities
│   └── main.py           # CLI entry point that runs the full pipeline
├── outputs/               # Saved model weights + loss curve plots (generated)
├── data/                    # (optional) local dataset cache
├── assets/                 # Placeholder folder — drop your result screenshots here (see Results)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/anime-cnn-classifier.git
cd anime-cnn-classifier
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If you don't have a GPU, the CPU-only PyTorch wheel works fine for this small dataset:
> `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu`

## Usage

Run the full pipeline (download data, train, plot loss, visualize predictions):

```bash
python src/main.py
```

### Useful options

```bash
python src/main.py --epochs 10 --activation leaky_relu
python src/main.py --no-visualize          # skip matplotlib image grids (e.g. for headless/CI runs)
python src/main.py --classes anastasia takao --zip-url <your-zip-url>
```

Run `python src/main.py --help` to see all available flags.

### Training only (no visualization), as a script

```bash
python src/train.py
```

This uses the default Anastasia/Takao dataset and saves the trained model to
`outputs/anime_cnn.pth` and the loss curve to `outputs/loss_curve.png`.

## How it works

1. **Data loading** (`dataset.py`) — downloads a zip file of `.jpg` images,
   buckets them by class based on filename prefix, and wraps them in a
   PyTorch `Dataset` (`AnimeDataset`) that resizes images to 64x64 and
   normalizes them to `[-1, 1]`.
2. **Model** (`model.py`) — a small CNN:
   `Conv(3→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → FC(128) → FC(num_classes)`.
3. **Training** (`train.py`) — 80/20 train/validation split via
   `SubsetRandomSampler`, `CrossEntropyLoss`, and the `Adam` optimizer.
4. **Evaluation** (`evaluate.py`) — visualizes a batch of validation images
   alongside their actual vs. predicted labels.

## Results

> 📁 **To make these images show up:** drop your screenshots into the
> `assets/` folder using the exact filenames below, then `git add assets/ && git commit -m "Add result screenshots" && git push`.
> GitHub will render them automatically since the README already links to these paths.
>
> | Filename (place in `assets/`) | What it should be |
> |---|---|
> | `loss_curve_baseline_5epoch.png` | Train/val loss curve from the default 5-epoch run |
> | `predictions_validation_grid.png` | Validation batch grid showing Actual vs. Predicted labels |
> | `loss_curve_10epoch_exercise2.png` | Train/val loss curve from the 10-epoch Exercise 2 run |
> | `dataset_exercise3_new_characters.png` | Sample grid of the new character pair from Exercise 3 |

**Baseline run** — default `relu` activation, 5 epochs, Anastasia vs. Takao:

![Baseline loss curve](assets/loss_curve_baseline_5epoch.png)

The model converges quickly on this small dataset, with both training and
validation loss approaching zero within the first 3 epochs.

Predictions on a batch from the validation set — every image is classified
correctly (`0` = Anastasia, `1` = Takao):

![Validation predictions](assets/predictions_validation_grid.png)

**Exercise 2 — more epochs (10 instead of 5):**

![10-epoch loss curve](assets/loss_curve_10epoch_exercise2.png)

Training for longer doesn't meaningfully improve validation loss here — it
was already near zero by epoch 3 — so the extra epochs mostly just continue
driving training loss down (a sign you're approaching the point of
diminishing returns / overfitting risk on a dataset this small).

**Exercise 3 — a different character pair** (`arcueid_brunestud` vs.
`yukinoshita_yukino`, 50 images each):

![New dataset sample](assets/dataset_exercise3_new_characters.png)

Swapping in a visually distinct character pair (blonde/red-eyed vs.
dark-haired) is a good sanity check that the pipeline generalizes — you can
reproduce this with:

```bash
python src/main.py --classes arcueid_brunestud yukinoshita_yukino \
    --zip-url https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/yNB99OssvDWOrNTHf2Yuxw/data-practice.zip
```

## Extending this project

- **Use your own dataset**: point `--zip-url` at a zip of `.jpg` images and
  pass the corresponding `--classes` (must match filename prefixes).
- **More classes**: `AnimeDataset` and `AnimeCNN` both support an arbitrary
  number of classes — just pass a longer `classes` list.
- **Different activation**: pass `--activation leaky_relu`.
- **Larger images**: pass `--image-size`, but note `model.py`'s `fc1` layer
  assumes a 64×64 input (after two 2x2 pools, the feature map is 16×16).
  Update `fc1`'s input size to `64 * (image_size // 4) ** 2` if you change this.

## Acknowledgements

Adapted from an IBM Skills Network educational notebook on CNNs for anime
image classification. Original dataset/paper: AniWho (Common, 2022).

## License

MIT — see [LICENSE](LICENSE).
