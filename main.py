import argparse

from models.fscil_trainer import FSCILTrainer
from utils import *

# C:\Users\qinzh\PycharmProjects\TFG\checkpoint\mini_imagenet\session0_max_acc_11641.pth
# C:\Users\qinzh\PycharmProjects\TFG\checkpoint\cifar100\session0_max_acc_11599.pth
# C:\Users\qinzh\PycharmProjects\TFG\checkpoint\cub200\session0_max_acc_11626.pth
# MODEL_DIR=r'C:\Users\qinzh\PycharmProjects\TFG\checkpoint\cifar100\session0_max_acc_11599.pth'
MODEL_DIR = None
DATA_DIR = 'D:/fscil_lmu/data/'


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=150)  ## for code test set to 1 default: 100
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_gan', type=float, default=0.0001)
    parser.add_argument('-beta', type=float, default=0.5)
    parser.add_argument('-decay', type=float, default=0.001)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)
    parser.add_argument('-output_length', type=int, default=256)
    parser.add_argument('-hidden_length', type=int, default=1024)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0,
                        help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)

    # for episode learning
    parser.add_argument('-train_episode', type=int, default=100)  ## for code test set to 1 default: 50
    parser.add_argument('-episode_shot', type=int, default=5)
    parser.add_argument('-generate_shot', type=int, default=5)
    parser.add_argument('-episode_way', type=int, default=5)
    parser.add_argument('-episode_query', type=int, default=10)

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-use_gpu', type=bool, default=True)

    return parser


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = 'D:\\fscil_lmu\\pretrain'
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    if args.use_gpu:
        args.num_gpu = set_gpu(args)

    trainer = FSCILTrainer(args)
    # trainer=FSCILTrainer(args)
    trainer.train()
