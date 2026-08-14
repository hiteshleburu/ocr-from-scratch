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
| `detection` | In Progress | Locates text regions within a full screenshot |
| `entities` | Planned | Classifies recognized text as sensitive (name, email, phone, etc.) |
| `redaction` | Planned | Blurs/masks the identified sensitive regions |

Each stage lives under `src/<stage>/`, with its own training script and checkpoints.

## Structure

```
ocr-from-scratch/
├── notebooks/              # step-by-step build notebooks
├── src/
│   └── recognition/
│       ├── vocab.py           # character tokenizer
│       ├── dataset.py          # Dataset, transforms, collate
│       ├── model.py             # CNN encoder + transformer decoder
│       ├── metrics.py            # word accuracy, CER
│       ├── generate.py            # autoregressive inference
│       ├── train.py                # training script
│       └── vocab.json               # saved character vocabulary
├── data/synth/              # synthetic data (not tracked in git)
├── checkpoints/               # trained weights (not tracked in git)
└── runs.csv                     # log of every training run
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
    --labels_path data/synth/train_v2/labels.txt \
    --image_dir data/synth/train_v2 \
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
| `--log_path` | `runs.csv` | Where run results are logged |

Full list of arguments in `src/recognition/train.py`.

Every run saves its best checkpoint (by validation loss), its exact validation split
(for reproducible eval later), and logs config + results as a row in `runs.csv`.

### Results

Current baseline: 50,000 synthetic images, 3-layer decoder, `d_model=256`, 30 epochs, cosine LR decay.

| Metric | Score |
|---|---|
| Overall word accuracy | 96.4% |
| Dictionary word accuracy | 99.9% |
| Random string accuracy | 88.1% |
| Character error rate | 1.36% |

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
