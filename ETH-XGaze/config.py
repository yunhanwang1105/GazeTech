import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--data_dir', type=str, default='/tudelft.net/staff-umbrella/NightId/eth-xgaze/raw/multi-region_3_5',
                      help='Directory of the data')
data_arg.add_argument('--batch_size', type=int, default=100,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=16,
                      help='# of subprocesses to use for data loading')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=30, # Note: we train poolformer for 50 epochs
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.0001,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--lr_decay_factor', type=float, default=0.1,
                       help='How much (factor) to decay lr')

train_arg.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                       help='Number of epochs to warm up, if none, then do not warm up')
train_arg.add_argument('--continue_train', type=str2bool, default=False,
                       help='Whether to load checkpoint and continue training')
train_arg.add_argument('--contiue_train_model_path', type=str, default='./ckpt/epoch_20_ckpt.pth.tar',
                       help='Directory in which to contiue training on')

# model building
train_arg.add_argument('--model_name', type=str, default='face_res50',
                       help='The name of the model to train: face_res50, multi_region_res50, multi_region_res50_share_eyenet, face_poolformer24')
                        # change no. epochs to 50 if use poolformer
train_arg.add_argument('--in_stride', type=int, default=2,
                       help='The stride of the first conv-layer of CNN or the first patch embedding layer of transformer')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--pre_trained_model_path', type=str, default='./ckpt/epoch_24_ckpt.pth.tar',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--print_freq', type=int, default=1000,
                      help='How frequently to print training details')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt',
                      help='Directory in which to save model checkpoints')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
