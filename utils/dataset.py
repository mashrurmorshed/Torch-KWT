import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import functools
import librosa
import glob
import os
from tqdm import tqdm
import multiprocessing as mp

from utils.augment import time_shift, resample, spec_augment
from audiomentations import AddBackgroundNoise


LABEL_LIST = None
LABEL_2_IDX = None
IDX_2_LABEL = None


def get_train_val_test_split(root: str, val_file: str, test_file: str):
    """Creates train, val, and test split according to provided val and test files.

    Args:
        root (str): Path to base directory of the dataset.
        val_file (str): Path to file containing list of validation data files.
        test_file (str): Path to file containing list of test data files.
    
    Returns:
        train_list (list): List of paths to training data items.
        val_list (list): List of paths to validation data items.
        test_list (list): List of paths to test data items.
    """
    
    ####################
    # Labels
    ####################

    global LABEL_LIST, LABEL_2_IDX, IDX_2_LABEL
    LABEL_LIST = [label for label in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, label)) and label[0] != "_"]
    LABEL_2_IDX = {label: idx for idx, label in enumerate(LABEL_LIST)}
    IDX_2_LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

    ###################
    # Split
    ###################

    all_files_set = set()
    for label in LABEL_LIST:
        all_files_set.update(set(glob.glob(os.path.join(root, label, "*.wav"))))
    
    with open(val_file, "r") as f:
        val_files_set = set(map(lambda a: os.path.join(root, a), f.read().rstrip("\n").split("\n")))
    
    with open(test_file, "r") as f:
        test_files_set = set(map(lambda a: os.path.join(root, a), f.read().rstrip("\n").split("\n"))) 
    
    assert len(val_files_set.intersection(test_files_set)) == 0, "Sanity check: No files should be common between val and test."
    
    all_files_set -= val_files_set
    all_files_set -= test_files_set
    
    train_list = list(all_files_set)
    val_list = list(val_files_set)
    test_list = list(test_files_set)
    
    print(f"Number of training samples: {len(train_list)}")
    print(f"Number of validation samples: {len(val_list)}")
    print(f"Number of test samples: {len(test_list)}")

    return train_list, val_list, test_list


class GoogleSpeechDataset(Dataset):
    """Dataset wrapper for Google Speech Commands V2."""
    
    def __init__(self, data_list: list, audio_settings: dict, aug_settings: dict = None, cache: bool = False):
        super().__init__()

        self.audio_settings = audio_settings
        self.aug_settings = aug_settings
        self.cache = cache

        if cache:
            print("Caching dataset into memory.")
            self.data_list = init_cache(data_list, audio_settings["sr"])
        else:
            self.data_list = data_list
            
        # labels
        global LABEL_2_IDX
        self.label_list = []
        for path in data_list:
            self.label_list.append(LABEL_2_IDX[path.split("/")[-2]])

        
        if aug_settings is not None:
            if "bg_noise" in self.aug_settings:
                self.bg_adder = AddBackgroundNoise(sounds_path=aug_settings["bg_noise"]["bg_folder"])


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        if self.cache:
            x = self.data_list[idx]
        else:
            x = librosa.load(self.data_list[idx], self.audio_settings["sr"])[0]

        x = self.transform(x)
        label = self.label_list[idx]

        return x, label


    def transform(self, x):
        """Applies necessary preprocessing to audio.

        Args:
            x (np.ndarray) - Input waveform; array of shape (n_samples, ).
        
        Returns:
            x (torch.FloatTensor) - MFCC matrix of shape (n_mfcc, T).
        """

        sr = self.audio_settings["sr"]

        ###################
        # Waveform 
        ###################

        if self.aug_settings is not None:
            if "bg_noise" in self.aug_settings:
                x = self.bg_adder(samples=x, sample_rate=sr)

            if "time_shift" in self.aug_settings:
                x = time_shift(x, sr, **self.aug_settings["time_shift"])

            if "resample" in self.aug_settings:
                x, _ = resample(x, sr, **self.aug_settings["resample"])
            
        x = librosa.util.fix_length(x, sr)

        ###################
        # Spectrogram
        ###################
     
        x = librosa.feature.melspectrogram(y=x, **self.audio_settings)        
        x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=self.audio_settings["n_mels"])

        if self.aug_settings is not None:
            if "spec_aug" in self.aug_settings:
                x = spec_augment(x, **self.aug_settings["spec_aug"])

        x = torch.from_numpy(x).float().unsqueeze(0)
        return x



def init_cache(data_list: list, sr: int, n_cache_workers: int = 4) -> list:
    """Loads entire dataset into memory for later use.

    Args:
        data_list (list): List of data items.
        sr (int): Sampling rate.
        n_cache_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        cache (list): List of data items.
    """

    cache = []
    loader_fn = functools.partial(librosa.load, sr=sr)

    pool = mp.Pool(n_cache_workers)

    for audio, sr in tqdm(pool.imap(func=loader_fn, iterable=data_list), total=len(data_list)):
        cache.append(audio)
    
    pool.close()
    pool.join()

    return cache


def get_loader(data_list, config, train=True):
    """Creates dataloaders for training, validation and testing.

    Args:
        config (dict): Dict containing various settings for the training run.
        train (bool): Training or evaluation mode.
        
    Returns:
        dataloader (DataLoader): DataLoader wrapper for training/validation/test data.
    """
    
    dataset = GoogleSpeechDataset(
        data_list=data_list,
        audio_settings=config["hparams"]["audio"],
        aug_settings=config["hparams"]["augment"] if train else None,
        cache=config["exp"]["cache"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
        shuffle=True if train else False
    )

    return dataloader

    