# PMMamba 

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
