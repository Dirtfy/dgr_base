import os.path
import copy
import random

import torch
from torch import optim
from torch import nn
import utils
import visual


def train(scholar, train_datasets, test_datasets, replay_mode, use_gan,
          label_schedule_list=None, 
          generate_ratio=0.5,
          generator_lambda=10.,
          generator_c_updates_per_g_update=5,
          generator_epochs=100,
          solver_epochs=50,
          importance_of_new_task=.5,
          batch_size=32,
          test_size=1024,
          sample_size=36,
          lr=1e-03, weight_decay=1e-05,
          beta1=.5, beta2=.9,
          loss_log_interval=30,
          eval_log_interval=50,
          image_log_interval=100,
          sample_log_interval=300,
          sample_log=False,
          sample_dir='./samples',
          checkpoint_dir='./checkpoints',
          generator_log_path=os.path.join('.', "log", "generator.txt"),
          solver_log_path=os.path.join('.', "log", "solver.txt"),
          collate_fn=None,
          cuda=False):
    # define solver criterion and generators for the scholar model.
    solver_criterion = nn.CrossEntropyLoss()
    solver_optimizer = optim.Adam(
        scholar.solver.parameters(),
        lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
    )
    if use_gan:
        generator_g_optimizer = optim.Adam(
            scholar.generator.generator.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )
        generator_c_optimizer = optim.Adam(
            scholar.generator.critic.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )
    else:
        generator_optimizer = optim.Adam(
            scholar.generator.parameters(),
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )

    # set the criterion, optimizers, and training configurations for the
    # scholar model.
    scholar.solver.set_criterion(solver_criterion)
    scholar.solver.set_optimizer(solver_optimizer)
    if use_gan:
        scholar.generator.set_lambda(generator_lambda)
        scholar.generator.set_generator_optimizer(generator_g_optimizer)
        scholar.generator.set_critic_optimizer(generator_c_optimizer)
        scholar.generator.set_critic_updates_per_generator_update(
            generator_c_updates_per_g_update
        )
    else:
        scholar.generator.optimizer = generator_optimizer
        scholar.generator.classifier_optimizer = solver_optimizer

    scholar.train()

    # define the previous scholar who will generate samples of previous tasks.
    previous_scholar = None
    previous_datasets = None

    for task, train_dataset in enumerate(train_datasets, 1):
        label_pool = []
        for schedule in label_schedule_list[:task]:
            label_pool.extend(schedule)
        # define callbacks for visualizing the training process.
        generator_training_callbacks = [_generator_training_callback(
            loss_log_interval=loss_log_interval,
            image_log_interval=image_log_interval,
            sample_log_interval=sample_log_interval,
            sample_log=sample_log,
            sample_dir=sample_dir,
            sample_size=sample_size,
            current_task=task,
            total_tasks=len(train_datasets),
            total_epochs=generator_epochs,
            batch_size=batch_size,
            replay_mode=replay_mode,
            log_path=generator_log_path,
            label_pool=label_pool,
            env=scholar.name,
        )]
        solver_training_callbacks = [_solver_training_callback(
            loss_log_interval=loss_log_interval,
            eval_log_interval=eval_log_interval,
            current_task=task,
            total_tasks=len(train_datasets),
            total_epochs=solver_epochs,
            batch_size=batch_size,
            test_size=test_size,
            test_datasets=test_datasets,
            replay_mode=replay_mode,
            sample_dir=sample_dir,
            cuda=cuda,
            collate_fn=collate_fn,
            log_path=solver_log_path,
            env=scholar.name,
        )]

        # train the scholar with generative replay.
        scholar.train_with_replay(
            train_dataset,
            scholar=previous_scholar,
            previous_datasets=previous_datasets,
            generate_folder_path=os.path.join(".", "sample", "train", f"{scholar.name}", f"task_{task}"),
            generate_ratio=generate_ratio,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            generator_epochs=generator_epochs,
            generator_training_callbacks=generator_training_callbacks,
            solver_epochs=solver_epochs,
            solver_training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
        )

        if label_schedule_list is not None:
            scholar.generator.label_pool.extend(label_schedule_list[task-1])
            print(f"label pool extend task_{task} label {label_schedule_list[task-1]} -> {scholar.generator.label_pool}")

        previous_scholar = (
            copy.deepcopy(scholar) if replay_mode == 'generative-replay' else
            None
        )
        previous_datasets = (
            train_datasets[:task] if replay_mode == 'exact-replay' else
            None
        )

    # save the model after the experiment.
    print()
    utils.save_checkpoint(scholar, checkpoint_dir)
    print()
    print()


def _generator_training_callback(
        loss_log_interval,
        image_log_interval,
        sample_log_interval,
        sample_log,
        sample_dir,
        current_task,
        total_tasks,
        total_epochs,
        batch_size,
        sample_size,
        replay_mode,
        log_path,
        label_pool,
        env):

    def cb(generator, progress, epoch, iteration, total_iter, result):
        desc = (
            '<Training Generator> '
            'task: {task}/{tasks} | '
            'epoch: {epoch}/{epochs} | '
            'progress: [{trained}/{total}] ({percentage:.0f}%) | '
            'loss => '
            'g: {g_loss:.4} / '
            'w: {w_dist:.4}'
        ).format(
            task=current_task,
            tasks=total_tasks,
            epoch=epoch,
            epochs=total_epochs,
            trained=batch_size * iteration,
            total=batch_size * total_iter,
            percentage=(100.*iteration/total_iter),
            g_loss=result['g_loss'],
            w_dist=-result['c_loss'],
        )
        progress.set_description(desc)

        with open(log_path, 'a') as f:
            f.write(desc+'\n')

        # # log the losses of the generator.
        # if iteration % loss_log_interval == 0:
        #     visual.visualize_scalar(
        #         result['g_loss'], 'generator g loss', iteration, env=env
        #     )
        #     visual.visualize_scalar(
        #         -result['c_loss'], 'generator w distance', iteration, env=env
        #     )

        # # log the generated images of the generator.
        # if iteration % image_log_interval == 0:
        #     label = torch.tensor(
        #                 random.choices(label_pool, k=sample_size)
        #             ).to(next(generator.parameters()).device)
        #     visual.visualize_images(
        #         generator.sample(sample_size, label).data,
        #         'generated samples ({replay_mode})'
        #         .format(replay_mode=replay_mode), env=env,
        #     )

        # # log the sample images of the generator
        # if iteration % sample_log_interval == 0 and sample_log:
        #     utils.test_model(generator, sample_size, os.path.join(
        #         sample_dir,
        #         env + '-sample-logs',
        #         str(iteration)
        #     ), verbose=False)

    return cb


def _solver_training_callback(
        loss_log_interval,
        eval_log_interval,
        current_task,
        total_tasks,
        total_epochs,
        batch_size,
        test_size,
        test_datasets,
        sample_dir,
        cuda,
        replay_mode,
        collate_fn,
        log_path,
        env):

    def cb(solver, progress, epoch, iteration, total_iter, result):
        desc = (
            '<Training Solver>    '
            'task: {task}/{tasks} | '
            'epoch: {epoch}/{epochs} | '
            'progress: [{trained}/{total}] ({percentage:.0f}%) | '
            'loss: {loss:.4} | '
            'prec: {prec:.4}'
        ).format(
            task=current_task,
            tasks=total_tasks,
            epoch=epoch,
            epochs=total_epochs,
            trained=batch_size * iteration,
            total=batch_size * total_iter,
            percentage=(100.*iteration/total_iter),
            loss=result['loss'],
            prec=result['precision'],
        )
        progress.set_description(desc)

        with open(log_path, 'a') as f:
            f.write(desc+'\n')

        # # log the loss of the solver.
        # if iteration % loss_log_interval == 0:
        #     visual.visualize_scalar(
        #         result['loss'], 'solver loss', iteration, env=env
        #     )

        # # evaluate the solver on multiple tasks.
        # if iteration % eval_log_interval == 0:
        #     names = ['task {}'.format(i+1) for i in range(len(test_datasets))]
        #     precs = [
        #         utils.validate(
        #             solver, test_datasets[i], test_size=test_size,
        #             cuda=cuda, verbose=False, collate_fn=collate_fn,
        #         ) if i+1 <= current_task else 0 for i in
        #         range(len(test_datasets))
        #     ]
        #     title = 'precision ({replay_mode})'.format(replay_mode=replay_mode)
        #     visual.visualize_scalars(
        #         precs, names, title,
        #         iteration, env=env
        #     )

    return cb
