# Rigetti — Surface Defect Classifier

Mobilenet_V3 image classifier for industrial metal surface defects
(Crack · Hole · Normal · Rust · Scratch).

---

## Submission Focus

This project is built with a focus on the following evaluation criteria:

- **Code quality and structure**: Utilizes a modular `ml/` package to isolate the core pipeline. This provides a clear **separation of concerns** between data handling, model architecture, and training logic. From an engineering perspective, this structure allows for **seamless expansion** (e.g., adding new architectures in `model.py` or new data sources in `dataset.py`) and enables the core ML logic to be imported as a library into other services or inference APIs.
- **Clarity of approach**: Centralized configuration (`config.py`) and a standard, reproducible training pipeline.
- **Correctness of evaluation**: Stratified 80/20 train/val split, with a completely held-out test set and detailed confusion matrix/per-class metrics.
- **Practical engineering decisions**: Docker support for portable execution, resume-capable checkpoints, Mixed Precision (AMP), and a lightweight MobileNet-V3-Small architecture for efficient training.

---

## Project layout

```
Rigetti/
├── dataset/
│   ├── train/          # class sub-folders (crack, hole, normal, rust, scratch)
│   ├── test/           # held-out test set
│   └── metadata.csv
├── ml/
│   ├── config.py       # all hyperparameters
│   ├── dataset.py      # DefectDataset + DataLoader factory
│   ├── model.py        # model, train_one_epoch, evaluate, test, save/load
│   ├── train.py        # training entry-point
│   ├── test.py         # inference entry-point
│   ├── visualize_logs.py # training curve visualization
│   └── runs/           # created at runtime
│       └── <timestamp>/
│           ├── best.pth
│           ├── latest.pth
│           ├── metrics.csv
│           ├── training_curves.png # created by visualize_logs.py
│           ├── predictions.csv
│           ├── confusion_matrix.png
│           └── report.txt
├── predictions.csv     # copy written to project root after training
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart — local

```bash
# 1. Create a virtual environment (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
#    GPU host:
pip install -r requirements.txt
#    CPU-only host:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib Pillow numpy tqdm

# 3. Train
python ml/train.py

# 4. Visualize Training Curves
python ml/visualize_logs.py ml/runs/<timestamp>

# 5. Evaluate (uses most recent run automatically)
python ml/test.py

# 6. Evaluate a specific checkpoint / run
python ml/test.py --run-dir ml/runs/20260407_120000
python ml/test.py --checkpoint ml/runs/20260407_120000/best.pth
```

---

## Quickstart — Docker (CPU)

```bash
# Build
docker compose build

# Train
docker compose up train

# Evaluate
docker compose up infer
```

Run artefacts are written to `./ml/runs/` and `./predictions.csv` on the
host via bind-mounts, so nothing is lost when the container exits.

---

## Configuration

All settings live in `ml/config.py` (`Config` dataclass).

| Parameter    | Default | Description                            |
| ------------ | ------- | -------------------------------------- |
| `batch_size` | 32      | Images per batch                       |
| `lr`         | 1e-3    | Adam learning rate                     |
| `epochs`     | 100     | Maximum training epochs                |
| `patience`   | 10      | Early-stopping patience (val_loss)     |
| `val_split`  | 0.2     | Fraction of train/ used for validation |
| `use_amp`    | True    | Mixed precision (auto-disabled on CPU) |
| `seed`       | 42      | Random seed for reproducibility        |

---

## Outputs

| File                   | Location               | Description                           |
| ---------------------- | ---------------------- | ------------------------------------- |
| `metrics.csv`          | run dir                | Per-epoch train/val loss & accuracy   |
| `best.pth`             | run dir                | Checkpoint with lowest val_loss       |
| `latest.pth`           | run dir                | Checkpoint from last completed epoch  |
| `predictions.csv`      | run dir + project root | `image_path,predicted_label`          |
| `confusion_matrix.png` | run dir                | 5×5 confusion matrix                  |
| `report.txt`           | run dir                | Overall accuracy + per-class accuracy |

---

## Model

- **Architecture**: MobileNet_V3_Small (ImageNet pre-trained)
- **Head**: Adjusted for 5 classes.
- **Optimiser**: Adam, lr=0.001
- **Scheduler**: CosineAnnealingLR
- **Augmentation**: random flips, colour jitter (train only)
- **Mixed precision**: CUDA float16 via `torch.cuda.amp`
