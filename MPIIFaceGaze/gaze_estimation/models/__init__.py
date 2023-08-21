import importlib

import torch
import yacs.config


def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
    dataset_name = config.mode.lower()
    module = importlib.import_module(
        f'gaze_estimation.models.{dataset_name}.{config.model.name}')
    in_stride = config.model.in_stride
    model = module.gaze_network(in_stride)
    if config.model.saved_model is not None:
        checkpoint = torch.load(config.model.saved_model, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Load model from ', config.model.saved_model)
    device = torch.device(config.device)
    model.to(device)
    return model
