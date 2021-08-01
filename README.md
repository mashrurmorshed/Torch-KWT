# Torch-KWT
Unofficial PyTorch implementation of *Keyword Transformer: A Self-Attention Model for Keyword Spotting*. 

## Setup

```
pip install -r requirements.txt
```

## Training

Training is fairly straightforward. Only a path to a config file is required.
```
python train.py --conf path/to/config.yaml
```

Refer to the [example config](sample_configs/base_config.yaml) to see how the config file looks like, and see the [config explanation](docs/config_file_explained.md) for a rundown of the various config parameters.

## Weights & Biases

You can optionally log your training runs with [wandb](https://wandb.ai/site). You may provide a path to a file containing your API key, or simply provide it manually from the login prompt when your start your training.

## Pretrained Checkpoints

| Model Name | Test Accuracy | Link |
| ---------- | ------------- | ---- |
|    KWT-1   |     95.98     | [kwt1-v01.pth](https://drive.google.com/uc?id=1Pglq3kFy9BVFk-bPVsbNuX_fzMGJ5uwy&export=download) |