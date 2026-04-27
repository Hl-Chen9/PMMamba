# PMMamba RADDet

This repository contains the minimal RADDet fine-tuning and evaluation code for the Mamba-based radar detector.

## Included Code

- `main_finetune_mamba.py`: RADDet train/eval entry point.
- `engine_finetune.py`: training and validation loops.
- `models/mamba_raddet.py`: Mamba detector.
- `models/RADDet_finetune/`: RADDet dataset loader, YOLO head/loss, mAP evaluation, anchors, and config.
- `datasets/RADDet.py`: RADDet RAD tensor dataset utility for pretraining/data checks.
- `utils.py`: distributed training, logging, checkpoint, and optimizer helpers.
- `selective_scan_interface.py`: selective-scan interface used by the Mamba model.
- `run_finetune_mamba.sh`: runnable shell template.
- `benchmark_mamba_raddet.py`: latency/FPS/memory benchmark for `RadarMamba_fpn`.

Training outputs, checkpoints, datasets, logs, notebooks, and all non-RADDet code are ignored by `.gitignore`.

## Setup

Install PyTorch for your CUDA version first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The Mamba module requires matching CUDA selective-scan dependencies. If `selective_scan_cuda` or `causal_conv1d_cuda` is missing, install/build `mamba-ssm` and `causal-conv1d` for your CUDA/PyTorch version.

## Dataset

Edit `models/RADDet_finetune/config.json` before running:

```json
"train_set_dir": "./data/RADDet_author/train",
"test_set_dir": "./data/RADDet_author/test"
```

`data/` is ignored by Git, so it can be a local copy or a symlink.

If you use `datasets/RADDet.py` directly, edit `datasets/RADDet_config.json` in the same way.

## Run

Fine-tune:

```bash
GPUS=0 NPROC_PER_NODE=1 bash run_finetune_mamba.sh
```

Evaluate from a checkpoint:

```bash
GPUS=0 NPROC_PER_NODE=1 EVAL_ONLY=1 FINETUNE=/path/to/checkpoint.pth bash run_finetune_mamba.sh
```

Benchmark:

```bash
python benchmark_mamba_raddet.py
```

## Upload

The repository is configured so `git add .` only stages RADDet Mamba files:

```bash
git add .
git status --short
git commit -m "Initial RADDet Mamba release"
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
```
