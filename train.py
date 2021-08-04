from argparse import ArgumentParser
from config_parser import get_config

from utils.loss import LabelSmoothingLoss
from utils.opt import get_optimizer
from utils.scheduler import WarmUpLR, get_scheduler
from utils.trainer import train, evaluate
from utils.dataset import get_train_val_test_split, get_loader
from utils.misc import seed_everything, count_params, get_model, calc_step, log

import torch
from torch import nn
import numpy as np
import wandb

import os
import yaml
import random
import time


def training_pipeline(config):
    """Initiates and executes all the steps involved with model training.

    Args:
        config (dict) - Dict containing various settings for the training run.
    """

    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    
    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)
    
    #####################################
    # initialize training items
    #####################################

    # data
    with open(config["train_list_file"], "r") as f:
        train_list = f.read().rstrip().split("\n")
    
    with open(config["val_list_file"], "r") as f:
        val_list = f.read().rstrip().split("\n")

    trainloader = get_loader(train_list, config, train=True)
    valloader = get_loader(val_list, config, train=False)

    # model
    model = get_model(config["hparams"]["model"])
    model = model.to(config["hparams"]["device"])
    print(f"Created model with {count_params(model)} parameters.")

    # loss
    if config["hparams"]["l_smooth"]:
        criterion = LabelSmoothingLoss(num_classes=config["hparams"]["model"]["num_classes"], smoothing=config["hparams"]["l_smooth"])
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = get_optimizer(model, config["hparams"]["optimizer"])
    
    # lr scheduler
    schedulers = {
        "warmup": None,
        "scheduler": None
    }

    if config["hparams"]["scheduler"]["n_warmup"]:
        schedulers["warmup"] = WarmUpLR(optimizer, total_iters=len(trainloader) * config["hparams"]["scheduler"]["n_warmup"])

    if config["hparams"]["scheduler"]["scheduler_type"] is not None:
        total_iters = len(trainloader) * max(1, (config["hparams"]["scheduler"]["max_epochs"] - config["hparams"]["scheduler"]["n_warmup"]))
        schedulers["scheduler"] = get_scheduler(optimizer, config["hparams"]["scheduler"]["scheduler_type"], total_iters)
    

    #####################################
    # Training Run
    #####################################

    print("Initiating training.")
    train(model, optimizer, criterion, trainloader, valloader, schedulers, config)

    #####################################
    # Final Test
    #####################################

    with open(config["test_list_file"], "r") as f:
        test_list = f.read().rstrip().split("\n")

    testloader = get_loader(test_list, config, train=False)
    final_step = calc_step(config["hparams"]["n_epochs"] + 1, len(trainloader), len(trainloader) - 1)
   

    # evaluating the final state (last.pth)
    test_acc, test_loss = evaluate(model, criterion, testloader, config["hparams"]["device"])
    log_dict = {
        "test_loss_last": test_loss,
        "test_acc_last": test_acc
    }
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pth)
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pth"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    test_acc, test_loss = evaluate(model, criterion, testloader, config["hparams"]["device"])
    log_dict = {
        "test_loss_best": test_loss,
        "test_acc_best": test_acc
    }
    log(log_dict, final_step, config)


def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])


    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()
        
        elif os.environ.get("WANDB_API_KEY", False):
            print(f"Found API key from env variable.")
        
        else:
            wandb.login()
        
        
        with wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"]):
            training_pipeline(config)
    
    else:
        training_pipeline(config)



if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    args = parser.parse_args()

    main(args)