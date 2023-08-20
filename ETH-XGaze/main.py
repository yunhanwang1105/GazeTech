import torch
from trainer import Trainer
from config import get_config
from data_loader import get_train_loader, get_test_loader

import numpy as np
import wandb
import argparse
import configparser

def run(config):
    kwargs = {}
    if config.use_gpu:
        # ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        np.random.seed(0)
        kwargs = {'num_workers': config.num_workers}

    # logging with weights and bias
    wandb.init(project='project-name', mode="disabled")
    
    load_mode = get_load_mode(config)

    # instantiate data loaders
    if config.is_train:
        data_loader = get_train_loader(
            config.data_dir, config.batch_size, load_mode, is_shuffle=True,
            **kwargs
        )
    else:
        data_loader = get_test_loader(
            config.data_dir, config.batch_size, load_mode, is_shuffle=False,
            **kwargs
        )
    # instantiate trainer
    trainer = Trainer(config, data_loader, load_mode)

    # either train
    if config.is_train:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()

def get_load_mode(config):
    
    if config.model_name == "face_res50":
        return "load_single_face"
    elif config.model_name == "multi_region_res50":
        return "load_multi_region"
    elif config.model_name == "multi_region_res50_share_eyenet":
        return "load_multi_region"
    elif config.model_name == "face_poolformer24":
        return "load_single_face"
    else:
        raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")

if __name__ == '__main__':
    config, unparsed = get_config()
    run(config)
