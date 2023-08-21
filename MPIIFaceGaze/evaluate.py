#!/usr/bin/env python

import pathlib

import numpy as np
import torch
import tqdm

from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                             create_model)
from gaze_estimation.utils import compute_angle_error, load_config, save_config


def test(model, test_loader, config):
    model.eval()
    device = torch.device(config.device)
    if config.model.name == 'face_res50':
        load_mode = 'load_single_face'
    elif config.model.name == 'multi_region_res50':
        load_mode = 'load_multi_region'
    elif config.model.name == 'multi_region_res50_share_eyenet':
        load_mode = 'load_multi_region'
    else:
        raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")

    predictions = []
    gts = []
    with torch.no_grad():
        for images, gazes in tqdm.tqdm(test_loader):
            if load_mode == 'load_single_face':
                image = images['face'].to(device)
                gazes = gazes.to(device)
                outputs = model(image)
            elif load_mode == 'load_multi_region':
                image = images['face'].to(device)
                left = images['left_eye'].to(device)
                right = images['right_eye'].to(device)
                gazes = gazes.to(device)
                outputs = model(image, left, right)
            predictions.append(outputs.cpu())
            gts.append(gazes.cpu())

    predictions = torch.cat(predictions)
    gts = torch.cat(gts)
    angle_error = float(compute_angle_error(predictions, gts).mean())
    return predictions, gts, angle_error


def main():
    config = load_config()

    output_rootdir = pathlib.Path(config.test.output_dir)
    checkpoint_name = pathlib.Path(config.test.checkpoint).stem
    output_dir = output_rootdir / checkpoint_name
    output_dir.mkdir(exist_ok=True, parents=True)
    save_config(config, output_dir)

    test_loader = create_dataloader(config, is_train=False)

    model = create_model(config)
    checkpoint = torch.load(config.test.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    predictions, gts, angle_error = test(model, test_loader, config)

    print(f'The mean angle error (deg): {angle_error:.2f}')


if __name__ == '__main__':
    main()
