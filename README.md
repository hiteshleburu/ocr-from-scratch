# OCR From Scratch

A text recognition system built from first principles - hand-built transformer decoder,
CNN encoder, trained on synthetic word images. PyTorch is used for tensor ops and autograd;
the attention mechanism, masking, and decoder layers are implemented from scratch rather
than using `nn.Transformer`.

**End goal:** scan screenshots of emails/messages, detect named entities (names, emails,
phone numbers, etc.), and automatically blur them out.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project stages

| Stage | Status | Description |
|---|---|---|
| `recognition` | Done | Reads text from a cropped word image |
| `detection` | Done | Locates text regions within a full screenshot |
| `entities` | Planned | Classifies recognized text as sensitive (name, email, phone, etc.) |
| `redaction` | Planned | Blurs/masks the identified sensitive regions |

Each stage lives under `src/<stage>/`, with its own training script, checkpoints, and run log.

## Structure

```
ocr-from-scratch/
├── notebooks/              # step-by-step build notebooks
├── src/
│   ├── recognition/
│   │   ├── vocab.py           # character tokenizer
│   │   ├── dataset.py          # Dataset, transforms, collate
│   │   ├── model.py             # CNN encoder + transformer decoder
│   │   ├── metrics.py            # word accuracy, CER
│   │   ├── generate.py            # autoregressive inference
│   │   ├── train.py                # training script
│   │   └── vocab.json               # saved character vocabulary
│   └── detection/
│       ├── model.py             # CNN encoder-decoder (U-Net), skip connections
│       ├── dataset.py            # Dataset, aspect-ratio-preserving transform
│       ├── postprocess.py         # mask -> boxes, IoU, precision/recall/F1
│       └── train.py                # training script
├── data/                    # synthetic data (not tracked in git)
│   ├── recognition/synth/
│   └── detection/
├── checkpoints/               # trained weights (not tracked in git)
│   ├── recognition/
│   └── detection/
└── runs/                        # per-stage experiment logs
    ├── runs_recognition.csv
    └── runs_detection.csv
```

---

## Recognition

Reads text from a cropped word image. CNN encoder + hand-built transformer decoder.

### Files

| File | Purpose |
|---|---|
| `vocab.py` | Character-level tokenizer. Builds/saves/loads the char<->id vocab, encode/decode strings |
| `dataset.py` | `OCRDataset`, image transforms, collate function, label file loader |
| `model.py` | `CNNEncoder`, `Decoder` (multi-head attention, causal masking, decoder layers), `OCRModel` |
| `metrics.py` | `word_accuracy` (exact match) and `cer` (character error rate) |
| `generate.py` | Autoregressive generation loop, with optional n-gram repeat blocking |
| `train.py` | CLI training script — data loading, training loop, eval, checkpointing, run logging |

### Architecture

```
word image [32x128]
   -> CNN encoder (height -> 1, width -> 32)
   -> 32 feature vectors (one per vertical strip)
   -> transformer decoder (self-attn, cross-attn, FFN x3 layers)
   -> character sequence
```

### Usage

**Generate synthetic training data** — see `notebooks/02_data_generation.ipynb`.

**Train:**

```bash
python -m src.recognition.train \
    --labels_path data/recognition/synth/train_v2/labels.txt \
    --image_dir data/recognition/synth/train_v2 \
    --epochs 30 \
    --lr_schedule cosine
```

**Key arguments:**

| Flag | Default | Description |
|---|---|---|
| `--labels_path` | required | Path to `labels.txt` (tab-separated filename, label) |
| `--image_dir` | required | Directory containing the images |
| `--epochs` | 30 | Number of training epochs |
| `--batch_size` | 128 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--lr_schedule` | flat | `flat` (warmup then constant) or `cosine` (decay to 0) |
| `--d_model` | 256 | Transformer embedding dimension |
| `--n_layers` | 3 | Number of decoder layers |
| `--n_heads` | 8 | Number of attention heads |
| `--ngram_block` | None | Block repeated n-grams of this length during eval generation |
| `--seed` | 42 | Random seed (data split, init) |
| `--checkpoint_dir` | `checkpoints/recognition` | Where checkpoints are saved |
| `--log_path` | `runs/runs_recognition.csv` | Where run results are logged |

Full list of arguments in `src/recognition/train.py`.

Every run saves its best checkpoint (by validation loss), its exact validation split
(for reproducible eval later), and logs config + results as a row in `runs/runs_recognition.csv`.

### Results

Best checkpoint: 63,000 synthetic images, 60+ fonts, 4-layer decoder, `d_model=384`,
70 epochs, cosine LR decay, batch size 128.

| Metric | Score |
|---|---|
| Word accuracy | 94.6% |
| Character error rate | 1.70% |

Font-diverse training was prioritized over raw benchmark score — a narrower, 6-font,
50,000-image run reached 96.4% word accuracy / 88.1% on unseen random strings, but the
broader-font model generalizes noticeably better to real, varied-font text crops from the
detection pipeline (see `13_pipeline_test.ipynb`).

### Notebooks

| Notebook | Contents |
|---|---|
| `01_tokenizer_and_metrics` | Vocab, tokenizer, word accuracy / CER |
| `02_data_generation` | Synthetic word image rendering |
| `03_dataset_and_dataloader` | Dataset, collate, tgt_in/tgt_out construction |
| `04_encoder` | CNN encoder, shape verification |
| `05_decoder` | Multi-head attention, causal masking, decoder layers |
| `06_training_loop` | Training loop, overfit sanity check, train/val tracking |
| `07_inference_eval` | Autoregressive generation, accuracy/CER evaluation |

---

## Detection

Locates text regions within a full screenshot. A U-Net-style CNN encoder-decoder predicts
a per-pixel binary mask (text vs. background); bounding boxes are then extracted from the
mask via connected-component analysis.

### Files

| File | Purpose |
|---|---|
| `model.py` | `ConvBlock`, `Encoder`/`Decoder` (skip connections), `DetectionModel` |
| `dataset.py` | `DetectionDataset`, `DetectionTransform` (aspect-ratio-preserving resize + pad) |
| `postprocess.py` | `mask_to_boxes` (connected components), `iou`, `evaluate_boxes` (precision/recall/F1) |
| `train.py` | CLI training script — data loading, training loop, eval, checkpointing, run logging |

### Architecture

```
screenshot [1x256x256]
   -> encoder (3x ConvBlock + MaxPool, 256->128->64->32, channels 1->32->64->128)
   -> bottleneck
   -> decoder (3x ConvTranspose2d + skip concat + ConvBlock, 32->64->128->256)
   -> per-pixel mask logits [1x256x256]
   -> connected components -> bounding boxes
```

Trained with a combined binary cross-entropy + Dice loss, chosen to counter the class
imbalance between text and background pixels.

### Usage

**Generate synthetic training data** — see `notebooks/08_detection_data_generation.ipynb`.
Synthetic screenshots use two randomly chosen layouts (email-style header + paragraph, or
chat-style message bubbles), with font-proportional spacing and procedurally generated
background/text contrast.

**Train:**

```bash
python -m src.detection.train \
    --image_dir data/detection/train_v2/images \
    --mask_dir data/detection/train_v2/masks \
    --n_images 25000 \
    --epochs 20
```

**Key arguments:**

| Flag | Default | Description |
|---|---|---|
| `--image_dir` | required | Directory containing the screenshot images |
| `--mask_dir` | required | Directory containing the matching binary masks |
| `--n_images` | required | Number of image/mask pairs to use |
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 16 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--base_channels` | 32 | Encoder/decoder starting channel width (doubles each level) |
| `--img_h` / `--img_w` | 256 / 256 | Input resolution after resize |
| `--seed` | 42 | Random seed (data split, init) |
| `--checkpoint_dir` | `checkpoints/detection` | Where checkpoints are saved |
| `--log_path` | `runs/runs_detection.csv` | Where run results are logged |

Full list of arguments in `src/detection/train.py`.

Every run saves its best checkpoint (by validation loss), its exact validation split, and
logs config + box-level eval metrics as a row in `runs/runs_detection.csv`.

### Results

Best checkpoint: 25,000 synthetic images, `base_channels=32`, 20 epochs.

| Metric | Score |
|---|---|
| Best val loss | 0.0147 |
| Precision | 99.43% |
| Recall | 99.23% |
| F1 | 99.32% |

Scaling from an initial 5,000-image run (F1 98.0%) to 25,000 images substantially reduced
the model's main failure mode — occasionally merging two adjacent words or lines into a
single bounding box — though it still occurs occasionally under tight word spacing.

### Notebooks

| Notebook | Contents |
|---|---|
| `08_detection_data_generation` | Synthetic screenshot + mask generation |
| `09_detection_dataset` | Dataset, aspect-ratio-preserving resize/padding |
| `10_detection_model` | U-Net encoder/decoder architecture |
| `11_detection_training` | BCE+Dice loss, training loop, overfit sanity check |
| `12_detection_eval` | Mask -> box extraction, IoU-based precision/recall/F1 |
| `13_pipeline_test` | Full detection -> crop -> recognition pipeline, end to end |