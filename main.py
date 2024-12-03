#!/usr/bin/env python3
import argparse
import os.path
import numpy as np
import torch
from torch.utils.data import ConcatDataset

import cfd.cfd
import cfd.diffusion
import cfd.embedding
import cfd.unet
import cfd.utils
import cgd.cdg
import cgd.classifier
import cgd.classifier_guidance
import cgd.unet
import id2.classifier
import id2.classifier_guidance
import id2.id2
import id2.unet
import utils
from data import get_dataset, DATASET_CONFIGS, split_by_label
from train import train
from dgr import Scholar
from models import WGAN, CNN

import cgd
import cfd
import id2

from logger import FileLogger

parser = argparse.ArgumentParser(
    'PyTorch implementation of Deep Generative Replay'
)

parser.add_argument(
    '--experiment', type=str,
    choices=['permutated-mnist', 'svhn-mnist', 'mnist-svhn', 'mnist_ci', 'cifar10_ci'],
    default='cifar10_ci'
)
parser.add_argument('--mnist-permutation-number', type=int, default=5)
parser.add_argument('--mnist-permutation-seed', type=int, default=0)
parser.add_argument(
    '--replay-mode', type=str, default='generative-replay',
    choices=['exact-replay', 'generative-replay', 'none'],
)

parser.add_argument('--generator-lambda', type=float, default=10.)
parser.add_argument('--generator-z-size', type=int, default=100)
parser.add_argument('--generator-c-channel-size', type=int, default=64)
parser.add_argument('--generator-g-channel-size', type=int, default=64)
parser.add_argument('--solver-depth', type=int, default=5)
parser.add_argument('--solver-reducing-layers', type=int, default=3)
parser.add_argument('--solver-channel-size', type=int, default=1024)

parser.add_argument('--generator-c-updates-per-g-update', type=int, default=5)
parser.add_argument('--generator-epochs', type=int, default=50)
parser.add_argument('--solver-epochs', type=int, default=20)
parser.add_argument('--importance-of-new-task', type=float, default=.3)
parser.add_argument('--lr', type=float, default=1e-04)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-05)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--test-size', type=int, default=1024)
parser.add_argument('--sample-size', type=int, default=36)

parser.add_argument('--sample-log', action='store_true')
parser.add_argument('--sample-log-interval', type=int, default=300)
parser.add_argument('--image-log-interval', type=int, default=100)
parser.add_argument('--eval-log-interval', type=int, default=50)
parser.add_argument('--loss-log-interval', type=int, default=30)
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--sample-dir', type=str, default='./samples')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

parser.add_argument('--model-name', type=str)

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train', action='store_true')
main_command.add_argument('--test', action='store_false', dest='train')

def get_cgc_cgd(dataset_config):
    n_classes = dataset_config['classes']
    img_size = dataset_config['size']
    img_channels = dataset_config['channels']

    unet = cgd.unet.UNet(
        n_classes=n_classes,
        img_channels=img_channels)
    classifier = cgd.classifier.Classifier(
        n_classes=n_classes,
        img_channels=img_channels)

    device = torch.device("cuda")

    model = cgd.classifier_guidance.ClassifierGuidedDiffusion(
        unet=unet,
        classifier=classifier,
        img_size=img_size,
        image_channels=img_channels,
        device=device
    )
    
    return cgd.cdg.Classifier(
        model=classifier
    ), cgd.cdg.CDG(
        model=model,
        n_classes=n_classes
    ), False

def get_id2(dataset_config):
    n_classes = dataset_config['classes']
    img_size = dataset_config['size']
    img_channels = dataset_config['channels']

    unet = id2.unet.UNet(
        n_classes=n_classes,
        img_channels=img_channels)
    classifier = id2.classifier.Classifier(
        n_classes=n_classes,
        img_channels=img_channels)

    device = torch.device("cuda")

    model = id2.classifier_guidance.ClassifierGuidedDiffusion(
        unet=unet,
        classifier=classifier,
        img_size=img_size,
        image_channels=img_channels,
        device=device,
        cfg_uncondition_train_ratio=0.1
    )
    
    return id2.id2.Classifier(
        model=classifier
    ), id2.id2.Id2(
        model=model,
        n_classes=n_classes,
        lambda_cg=1,
        lambda_cfg=1,
        cg_cfg_ratio=0.5
    ), False

def get_cnn_cfd(args, dataset_config, c_emb_dim=10):
    n_classes = dataset_config['classes']
    img_size = dataset_config['size']
    img_channels = dataset_config['channels']

    device = torch.device("cuda")
    
    net = cfd.unet.Unet(
                in_ch = dataset_config['channels'],
                out_ch = dataset_config['channels'],
            ).to(device)
    
    cemblayer = cfd.embedding.ConditionalEmbedding(
        num_labels=dataset_config['classes'], 
        d_model=c_emb_dim, 
        dim=c_emb_dim).to(device)
    
    betas = cfd.utils.get_named_beta_schedule()

    model = cfd.diffusion.GaussianDiffusion(
                    dtype = torch.float32,
                    model = net,
                    betas = betas,
                    w = 3.0,
                    v = 1.0,
                    device = device
                ).to(device)
    
    return CNN(
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        classes=dataset_config['classes'],
        depth=args.solver_depth,
        channel_size=args.solver_channel_size,
        reducing_layers=args.solver_reducing_layers,
    ), cfd.cfd.CFD(
        model=model,
        embeder=cemblayer,
        img_shape=(img_channels, img_size, img_size),
        n_classes=n_classes
    ), False

def get_cnn_wgan(args, dataset_config):
    return CNN(
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        classes=dataset_config['classes'],
        depth=args.solver_depth,
        channel_size=args.solver_channel_size,
        reducing_layers=args.solver_reducing_layers,
    ), WGAN(
        z_size=args.generator_z_size,
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        c_channel_size=args.generator_c_channel_size,
        g_channel_size=args.generator_g_channel_size,
    ), True      



if __name__ == '__main__':
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda
    torch.set_default_dtype(torch.float32)
    experiment = args.experiment

    if experiment == 'permutated-mnist':
        # generate permutations for the mnist classification tasks.
        np.random.seed(args.mnist_permutation_seed)
        permutations = [
            np.random.permutation(DATASET_CONFIGS['mnist']['size']**2) for
            _ in range(args.mnist_permutation_number)
        ]

        # prepare the datasets.
        train_datasets = [
            get_dataset('mnist', permutation=p)
            for p in permutations
        ]
        test_datasets = [
            get_dataset('mnist', train=False, permutation=p)
            for p in permutations
        ]

        # decide what configuration to use.
        dataset_config = DATASET_CONFIGS['mnist']
    elif experiment in ('svhn-mnist', 'mnist-svhn'):
        mnist_color_train = get_dataset(
            'mnist-color', train=True
        )
        mnist_color_test = get_dataset(
            'mnist-color', train=False
        )
        svhn_train = get_dataset('svhn', train=True)
        svhn_test = get_dataset('svhn', train=False)

        # prepare the datasets.
        train_datasets = (
            [mnist_color_train, svhn_train] if experiment == 'mnist-svhn' else
            [svhn_train, mnist_color_train]
        )
        test_datasets = (
            [mnist_color_test, svhn_test] if experiment == 'mnist-svhn' else
            [svhn_test, mnist_color_test]
        )

        # decide what configuration to use.
        dataset_config = DATASET_CONFIGS['mnist-color']
    elif experiment == "mnist_ci":
        mnist_train = get_dataset(
            'mnist', train=True
        )
        mnist_test = get_dataset(
            'mnist', train=False
        )

        # path = os.path.join('.', "datasets", "mnist", "MNIST", "byLabel")
        # os.makedirs(path, exist_ok=True)
        splited_mnist_train = split_by_label(mnist_train)
        # print(splited_mnist_train)
        # for key, value in splited_mnist_train.items():
        #     torch.save(value, os.path.join(path, f"{key}-train"))

        splited_mnist_test = split_by_label(mnist_test)
        # print(splited_mnist_train)
        # for key, value in splited_mnist_test.items():
        #     torch.save(value, os.path.join(path, f"{key}-test"))

        label_list = list(splited_mnist_train.keys())
        schedule_list = [[0]]

        # selected, label_list = utils.sample(label_list, 4)
        # schedule_list.append(selected)

        # selected, label_list = utils.sample(label_list, 2)
        # schedule_list.append(selected)

        # selected, label_list = utils.sample(label_list, 2)
        # schedule_list.append(selected)
        
        # schedule_list.append(label_list)

        print(f"schedule list: {schedule_list}")

        # prepare the datasets.
        train_datasets = [
            ConcatDataset([
                splited_mnist_train[label]
                for label in schedule
            ]) for schedule in schedule_list
        ]
        test_datasets = [
            ConcatDataset([
                splited_mnist_test[label]
                for label in schedule
            ]) for schedule in schedule_list
        ]

        # decide what configuration to use.
        dataset_config = DATASET_CONFIGS['mnist']
    elif experiment == "cifar10_ci":
        cifar10_train = get_dataset(
            'cifar10', train=True
        )
        cifar10_test = get_dataset(
            'cifar10', train=False
        )

        splited_cifar10_train = split_by_label(cifar10_train)
        splited_cifar10_test = split_by_label(cifar10_test)

        label_list = list(splited_cifar10_train.keys())
        # schedule_list = [[2, 8, 9, 0], [5, 7], [1, 3], [6, 4]]
        schedule_list = [[0, 1], [2, 3], [4, 5], [6, 7]]
        # schedule_list = [[0], [1], [2]]
        # schedule_list = []

        # selected, label_list = utils.sample(label_list, 4)
        # schedule_list.append(selected)

        # selected, label_list = utils.sample(label_list, 2)
        # schedule_list.append(selected)

        # selected, label_list = utils.sample(label_list, 2)
        # schedule_list.append(selected)
        
        # schedule_list.append(label_list)

        print(f"schedule list: {schedule_list}")

        # prepare the datasets.
        train_datasets = [
            ConcatDataset([
                splited_cifar10_train[label]
                for label in schedule
            ]) for schedule in schedule_list
        ]
        test_datasets = [
            ConcatDataset([
                splited_cifar10_test[label]
                for label in schedule
            ]) for schedule in schedule_list
        ]

        # decide what configuration to use.
        dataset_config = DATASET_CONFIGS['cifar10']
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(experiment))

    # define the models.

    # solver, generator, use_gan = get_cnn_wgan(args, dataset_config)
    # solver, generator, use_gan = get_cgc_cgd(dataset_config=dataset_config)
    # solver, generator, use_gan = get_cnn_cfd(args=args, dataset_config=dataset_config)
    solver, generator, use_gan = get_id2(dataset_config=dataset_config)
    
    label = '{experiment}-{model_name}-{replay_mode}-r{importance_of_new_task}'.format(
        experiment=experiment,
        model_name=args.model_name,
        replay_mode=args.replay_mode,
        importance_of_new_task=(
            1 if args.replay_mode == 'none' else
            args.importance_of_new_task
        ),
    )
    scholar = Scholar(label, generator=generator, solver=solver)

    # initialize the model.
    utils.gaussian_intiailize(scholar, std=.02)

    # use cuda if needed
    if cuda:
        scholar.cuda()

    # determine whether we need to train the generator or not.
    train_generator = (
        args.replay_mode == 'generative-replay' or
        args.sample_log
    )

    logger = FileLogger(file_path=os.path.join(".","log"),file_name=label+".txt")
    logger.on()
    
    # run the experiment.
    if args.train:
        train(
            scholar, train_datasets, test_datasets,
            replay_mode=args.replay_mode,
            use_gan=use_gan,
            label_schedule_list=schedule_list,
            generate_ratio=0.01,
            generator_lambda=args.generator_lambda,
            generator_epochs=(
                args.generator_epochs if train_generator else 0
            ),
            generator_c_updates_per_g_update=(
                args.generator_c_updates_per_g_update
            ),
            solver_epochs=args.solver_epochs,
            importance_of_new_task=args.importance_of_new_task,
            batch_size=args.batch_size,
            test_size=args.test_size,
            sample_size=args.sample_size,
            lr=args.lr, weight_decay=args.weight_decay,
            beta1=args.beta1, beta2=args.beta2,
            loss_log_interval=args.loss_log_interval,
            eval_log_interval=args.eval_log_interval,
            image_log_interval=args.image_log_interval,
            sample_log_interval=args.sample_log_interval,
            sample_log=args.sample_log,
            sample_dir=args.sample_dir,
            checkpoint_dir=args.checkpoint_dir,
            collate_fn=utils.label_squeezing_collate_fn,
            cuda=cuda
        )
    else:
        path = os.path.join(args.sample_dir, '{}-sample'.format(scholar.name))
        utils.load_checkpoint(scholar, args.checkpoint_dir)
        utils.test_model(scholar.generator, args.sample_size, path)
    
    logger.off()
