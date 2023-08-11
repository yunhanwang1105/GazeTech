import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import os
import time
import numpy as np
import wandb

from utils import AverageMeter, angular_error

from warmup_scheduler import GradualWarmupScheduler

class Trainer(object):
    def __init__(self, config, data_loader, load_mode, val_loader=None, test_loader=None):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """

        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader
            self.num_train = len(self.train_loader.dataset)
            # for validation
            assert val_loader is not None
            self.val_loader = val_loader
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.batch_size = config.batch_size

        # training params
        self.epochs = config.epochs  # the total epoch to train
        self.start_epoch = 0
        self.lr = config.init_lr
        self.lr_patience = config.lr_patience
        self.lr_decay_factor = config.lr_decay_factor

        # misc params
        self.use_gpu = config.use_gpu
        self.ckpt_dir = config.ckpt_dir  # output dir
        self.print_freq = config.print_freq
        self.train_iter = 0
        self.pre_trained_model_path = config.pre_trained_model_path

        
        # build model (we support face_res50, multi_region_res50, multi_region_res50_share_eyenet)
        self.in_stride = config.in_stride
        self.model_name = config.model_name
        print("We are using: ", self.model_name, " with the stride of ", self.in_stride)

        self.model = None
        self.load_mode = load_mode

        if self.model_name == 'face_res50':
            from models.face_res50 import gaze_network
            self.model = gaze_network(in_stride=self.in_stride)
        elif self.model_name == 'multi_region_res50':
            from models.multi_region_res50 import gaze_network
            self.model = gaze_network(in_stride=self.in_stride)
        elif self.model_name == 'multi_region_res50_share_eyenet':
            from models.multi_region_res50_share_eyenet import gaze_network
            self.model = gaze_network(in_stride=self.in_stride)

        assert self.model is not None


        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(
            self.optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)

        if config.warmup_epochs:
            self.scheduler = GradualWarmupScheduler(
                                    self.optimizer,
                                    multiplier=1,
                                    total_epoch=config.warmup_epochs,
                                    after_scheduler=self.scheduler
                                )

        self.continue_train = config.continue_train

        if self.continue_train:
            self.contiue_train_model_path = config.contiue_train_model_path

        if self.use_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

    def train(self):
        print("\n[*] Train on {} samples".format(self.num_train))

        # Continue to train the model for another self.epochs
        if self.continue_train:
            print("Continue training on: ", self.contiue_train_model_path)
            self.load_checkpoint(is_strict=False, input_file_path=self.contiue_train_model_path)
            self.epochs = self.epochs + self.start_epoch 

        # train for each epoch
        for epoch in range(self.start_epoch, self.epochs):
            print(
                '\nEpoch: {}/{} - base LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.lr)
            )

            for param_group in self.optimizer.param_groups:
                print('Learning rate: ', param_group['lr'])

            # train for 1 epoch
            print('Now go to training')
            self.model.train()
            train_acc, loss_gaze = \
                self.train_one_epoch(epoch, self.train_loader)


            # save the model for each epoch
            add_file_name = 'epoch_' + str(epoch) + '_' + str(self.lr)
            # if epoch % 5 == 4:
            self.save_checkpoint(
                {'epoch': epoch + 1,
                'model_state': self.model.state_dict(),
                'optim_state': self.optimizer.state_dict(),
                'scheule_state': self.scheduler.state_dict()
                }, add=add_file_name
            )
            self.scheduler.step()  # update learning rate


    def train_one_epoch(self, epoch, data_loader, is_train=True):
        """
        Train the model for 1 epoch of the training set.
        """
        batch_time = AverageMeter()
        errors = AverageMeter()
        losses_gaze = AverageMeter()

        tic = time.time()
        for i, (input_img, target) in enumerate(data_loader):
            # depending on load mode, input differently
            if self.load_mode == 'load_single_face':
                input_var = torch.autograd.Variable(input_img['face'].float().cuda())
                pred_gaze = self.model(input_var)
            elif self.load_mode == 'load_multi_region':
                input_var = torch.autograd.Variable(input_img['face'].float().cuda())
                left_var = torch.autograd.Variable(input_img['left_eye'].float().cuda())
                right_var = torch.autograd.Variable(input_img['right_eye'].float().cuda())
                pred_gaze = self.model(input_var, left_var, right_var)

            target_var = torch.autograd.Variable(target.float().cuda())
            gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), target_var.cpu().data.numpy()))
            errors.update(gaze_error_batch.item(), input_var.size()[0])

            loss_gaze = F.l1_loss(pred_gaze, target_var)
            self.optimizer.zero_grad()
            loss_gaze.backward()
            self.optimizer.step()
            losses_gaze.update(loss_gaze.item(), input_var.size()[0])

            wandb.log({"loss": loss_gaze})
            wandb.log({"gaze_error": gaze_error_batch})

            # report information
            if i % self.print_freq == 0 and i != 0:
                print('--------------------------------------------------------------------')
                msg = "train error: {:.3f} - loss_gaze: {:.5f}"
                print(msg.format(errors.avg, losses_gaze.avg))

                # measure elapsed time
                print('iteration ', self.train_iter)
                toc = time.time()
                batch_time.update(toc - tic)
                # print('Current batch running time is ', np.round(batch_time.avg / 60.0), ' mins')
                tic = time.time()
                # estimate the finish time
                est_time = (self.epochs - epoch) * (self.num_train / self.batch_size) * batch_time.avg / 60.0
                print('Estimated training time left: ', np.round(est_time), ' mins')

                errors.reset()
                losses_gaze.reset()
            self.train_iter = self.train_iter + 1

        self.validate() # validate on valset for each epoch

        toc = time.time()
        batch_time.update(toc-tic)
        wandb.log({"training_epoch_gaze_error": errors.avg})

        print('running time is ', batch_time.avg)
        return errors.avg, losses_gaze.avg

    def validate(self):
        """
        Test the pre-treained model on the whole val set.
        """
        self.model.eval()
        errors = AverageMeter()

        for i, (input_img, target) in enumerate(self.val_loader):
            # depending on load mode, input differently
            if self.load_mode == 'load_single_face':
                input_var = torch.autograd.Variable(input_img['face'].float().cuda())
                pred_gaze = self.model(input_var)
            elif self.load_mode == 'load_multi_region':
                input_var = torch.autograd.Variable(input_img['face'].float().cuda())
                left_var = torch.autograd.Variable(input_img['left_eye'].float().cuda())
                right_var = torch.autograd.Variable(input_img['right_eye'].float().cuda())
                pred_gaze = self.model(input_var, left_var, right_var)
            target_var = torch.autograd.Variable(target.float().cuda())
            gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), target_var.cpu().data.numpy()))
            errors.update(gaze_error_batch.item(), input_var.size()[0])
        print('val_gaze_error:', errors.avg)
        wandb.log({'each_epoch_val_gaze_error': errors.avg})
        

    def test(self):
        """
        Test the pre-treained model on the whole test set. Note there is no label released to public, you can
        only save the predicted results. You then need to submit the test resutls to our evaluation website to
        get the final gaze estimation error.
        """
        print('We are now doing the final test')
        self.model.eval()
        self.load_checkpoint(is_strict=False, input_file_path=self.pre_trained_model_path)
        errors = AverageMeter()

        print('Testing on ', self.num_test, ' samples')
        for i, (input_img, target) in enumerate(self.test_loader):
            # depending on load mode, input differently
            if self.load_mode == 'load_single_face':
                input_var = torch.autograd.Variable(input_img['face'].float().cuda())
                pred_gaze = self.model(input_var)
            elif self.load_mode == 'load_multi_region':
                input_var = torch.autograd.Variable(input_img['face'].float().cuda())
                left_var = torch.autograd.Variable(input_img['left_eye'].float().cuda())
                right_var = torch.autograd.Variable(input_img['right_eye'].float().cuda())
                pred_gaze = self.model(input_var, left_var, right_var)
            target_var = torch.autograd.Variable(target.float().cuda())
            gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), target_var.cpu().data.numpy()))
            errors.update(gaze_error_batch.item(), input_var.size()[0])
            wandb.log({"gaze_error": gaze_error_batch})

        print('final gaze error:', errors.avg)

    def save_checkpoint(self, state, add=None):
        """
        Save a copy of the model
        """
        if add is not None:
            filename = add + '_ckpt.pth.tar'
        else:
            filename ='ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        print('save file to: ', filename)

    def load_checkpoint(self, input_file_path='./ckpt/ckpt.pth.tar', is_strict=True):
        """
        Load the copy of a model.
        """
        print('load the pre-trained model: ', input_file_path)
        ckpt = torch.load(input_file_path)

        # load variables from checkpoint
        self.model.load_state_dict(ckpt['model_state'], strict=is_strict)
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.scheduler.load_state_dict(ckpt['scheule_state'])
        ###
        self.start_epoch = ckpt['epoch']
        ###

        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                input_file_path, ckpt['epoch'])
        )
