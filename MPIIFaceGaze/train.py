#!/usr/bin/env python

import time

import torch
import torchvision.utils
from fvcore.common.checkpoint import Checkpointer

from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                             create_logger, create_loss, create_model,
                             create_optimizer, create_scheduler)
from gaze_estimation.utils import (AverageMeter, compute_angle_error,
                                   create_train_output_dir, load_config,
                                   save_config, set_seeds, setup_cudnn)


def train(epoch, model, optimizer, scheduler, loss_function, train_loader,
          config,  logger):
    logger.info(f'Train {epoch}')

    model.train()

    device = torch.device(config.device)

    if config.model.name == 'face_res50':
        load_mode = 'load_single_face'
    elif config.model.name == 'multi_region_res50':
        load_mode = 'load_multi_region'
    elif config.model.name == 'multi_region_res50_share_eyenet':
        load_mode = 'load_multi_region'
    else:
        raise Exception("Please enter a correct model name or choose a correct load mode for your model (load_single_face or load_multi_region).")

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    for step, (images, gazes) in enumerate(train_loader):
        if load_mode == 'load_single_face':
            image = images['face'].to(device)
            gazes = gazes.to(device)
            optimizer.zero_grad()
            outputs = model(image)
        elif load_mode == 'load_multi_region':
            image = images['face'].to(device)
            left = images['left_eye'].to(device)
            right = images['right_eye'].to(device)
            gazes = gazes.to(device)
            optimizer.zero_grad()
            outputs = model(image, left, right)


        loss = loss_function(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = image.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        if step % config.train.log_period == 0:
            logger.info(f'Epoch {epoch} '
                        f'Step {step}/{len(train_loader)} '
                        f'lr {scheduler.get_last_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'angle error {angle_error_meter.val:.2f} '
                        f'({angle_error_meter.avg:.2f})')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')


def validate(epoch, model, loss_function, val_loader, config,
             logger):
    logger.info(f'Val {epoch}')

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
    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, (images, gazes) in enumerate(val_loader):
            if load_mode == 'load_single_face':
                image = images['face'].to(device)
                gazes = gazes.to(device)
                optimizer.zero_grad()
                outputs = model(image)
            elif load_mode == 'load_multi_region':
                image = images['face'].to(device)
                left = images['left_eye'].to(device)
                right = images['right_eye'].to(device)
                gazes = gazes.to(device)
                optimizer.zero_grad()
                outputs = model(image, left, right)
            loss = loss_function(outputs, gazes)

            angle_error = compute_angle_error(outputs, gazes).mean()

            num = image.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

    logger.info(f'Epoch {epoch} '
                f'loss {loss_meter.avg:.4f} '
                f'angle error {angle_error_meter.avg:.2f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')


def main():
    config = load_config()

    set_seeds(config.train.seed)
    setup_cudnn(config)

    output_dir = create_train_output_dir(config)
    save_config(config, output_dir)
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')

    train_loader, val_loader = create_dataloader(config, is_train=True)
    model = create_model(config)
    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir.as_posix(),
                                save_to_disk=True)


    if config.train.val_first:
        validate(0, model, loss_function, val_loader, config,
                 logger)

    for epoch in range(config.model.start_epoch+1, config.scheduler.epochs + 1):
        train(epoch, model, optimizer, scheduler, loss_function, train_loader,
              config, logger)
        scheduler.step()

        if epoch % config.train.val_period == 0:
            validate(epoch, model, loss_function, val_loader, config,
                     logger)

        if (epoch % config.train.checkpoint_period == 0
                or epoch == config.scheduler.epochs):
            checkpoint_config = {'epoch': epoch, 'config': config.as_dict()}
            checkpointer.save(f'checkpoint_{epoch:04d}', **checkpoint_config)


if __name__ == '__main__':
    main()
