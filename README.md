# Torch-KWT
Unofficial PyTorch implementation of *Keyword Transformer: A Self-Attention Model for Keyword Spotting*.

<a href="#" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

## Setup

```
pip install -r requirements.txt
```

## Dataset
To download the Google Speech Commands V2 dataset, you may run the provided bash script as below. This would download and extract the dataset to the "destination path" provided.

```
sh ./download_gspeech_v2.sh <destination_path>
```

## Training

The Speech Commands V2 dataset provides a "validation_list.txt" file and a "testing_list.txt" file. Run:

```
python make_data_list.py -v <path/to/validation_list.txt> -t <path/to/testing_list.txt> -d <path/to/dataset/root> -o <output dir>
```

This will create the files `training_list.txt`, `validation_list.txt`, `testing_list.txt` and `label_map.json` at the specified output dir. These will be needed for training.

Running `train.py` is fairly straightforward. Only a path to a config file is required.
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
