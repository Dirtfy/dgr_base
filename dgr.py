import abc

import os

import random
from itertools import cycle

import utils
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import DataLoader

import data


# ============
# Base Classes
# ============

class GenerativeMixin(object):
    """Mixin which defines a sampling iterface for a generative model."""
    def sample(self, batch_size, y):
        raise NotImplementedError


class BatchTrainable(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class which defines a generative-replay-based training
    interface for a model.

    """
    @abc.abstractmethod
    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        raise NotImplementedError


# ==============================
# Deep Generative Replay Modules
# ==============================

class Generator(GenerativeMixin, BatchTrainable):
    """Abstract generator module of a scholar module"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_pool = []


class Solver(BatchTrainable):
    """Abstract solver module of a scholar module"""
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.criterion = None

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def solve(self, x):
        scores = self(x)
        _, predictions = torch.max(scores, 1)
        return predictions

    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()

        # clear gradients.
        batch_size = x.size(0)
        self.optimizer.zero_grad()

        # run the model on the real data.
        real_scores = self.forward(x)
        real_loss = self.criterion(real_scores, y)
        _, real_predicted = real_scores.max(1)
        real_prec = (y == real_predicted).sum().data.item() / batch_size

        # run the model on the replayed data.
        if x_ is not None and y_ is not None:
            replay_scores = self.forward(x_)
            replay_loss = self.criterion(replay_scores, y_)
            _, replay_predicted = replay_scores.max(1)
            replay_prec = (y_ == replay_predicted).sum().data.item() / batch_size

            # calculate joint loss of real data and replayed data.
            loss = (
                importance_of_new_task * real_loss +
                (1-importance_of_new_task) * replay_loss
            )
            precision = (real_prec + replay_prec) / 2
        else:
            loss = real_loss
            precision = real_prec

        loss.backward()
        self.optimizer.step()
        return {'loss': loss.data.item(), 'precision': precision}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion


class Scholar(GenerativeMixin, nn.Module):
    """Scholar for Deep Generative Replay"""
    def __init__(self, label, generator, solver):
        super().__init__()
        self.label = label
        self.generator = generator
        self.solver = solver

    def train_with_replay(
            self, dataset, scholar=None, previous_datasets=None,
            generate_folder_path=None,
            generate_ratio=0.5,
            importance_of_new_task=.5, batch_size=32,
            generator_epochs=100,
            generator_training_callbacks=None,
            solver_epochs=50,
            solver_training_callbacks=None,
            collate_fn=None):
        # scholar and previous datasets cannot be given at the same time.
        mutex_condition_infringed = all([
            scholar is not None,
            bool(previous_datasets)
        ])
        assert not mutex_condition_infringed, (
            'scholar and previous datasets cannot be given at the same time'
        )

        # train the generator of the scholar.
        self._train_batch_trainable_with_replay(
            self.generator, dataset, scholar,
            previous_datasets=previous_datasets,
            generate_folder_path=generate_folder_path,
            generate_ratio=generate_ratio,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            epochs=generator_epochs,
            training_callbacks=generator_training_callbacks,
            collate_fn=collate_fn,
        )

        # train the solver of the scholar.
        self._train_batch_trainable_with_replay(
            self.solver, dataset, scholar,
            previous_datasets=previous_datasets,
            generate_folder_path=generate_folder_path,
            generate_ratio=generate_ratio,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            epochs=solver_epochs,
            training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
        )

    @property
    def name(self):
        return self.label

    def sample(self, batch_size, y):
        x = self.generator.sample(batch_size)
        # y = self.solver.solve(x)
        return x.data, y.data
    
    def generate_dataset(self, scholar, batch_size, total_size, folder_path) -> Dataset:

        os.makedirs(folder_path)

        assert total_size//batch_size > 0

        file_index = 0
        batch_schedule = [batch_size] * (total_size//batch_size) + [total_size%batch_size]
        progress = tqdm(batch_schedule, desc=f"generating dataset | folder_path: {folder_path}")
        for batch in progress:
            label = torch.tensor(
                        random.choices(scholar.generator.label_pool, k=batch)
                    )
                
            x, y = scholar.sample(batch, label)

            for image, label in zip(x, y):
                file_name = f"{file_index}_{label}.png"
                data.save_as_image(image, os.path.join(folder_path, file_name))

                file_index += 1

        return data.ImageFolderDataset(folder_path=folder_path)
    
    def _train_batch_trainable_with_replay(
            self, trainable, dataset, scholar=None,
            previous_datasets=None, generate_folder_path=None,
            generate_ratio=0.5,
            importance_of_new_task=.5, batch_size=32, epochs=100,
            training_callbacks=None, collate_fn=None):
        # do not train the model when given non-positive epochs.
        if epochs <= 0:
            return

        # create data loaders.
        data_loader = iter(utils.get_data_loader(
            dataset, batch_size, cuda=self._is_on_cuda(),
            collate_fn=collate_fn,
        ))

        is_generated = False
        if not previous_datasets and scholar is not None:
            if os.path.isdir(generate_folder_path):
                previous_dataset = data.ImageFolderDataset(generate_folder_path)
            else:
                previous_dataset = self.generate_dataset(
                    scholar, 
                    batch_size, 
                    int(len(dataset)*generate_ratio), 
                    generate_folder_path)
            previous_datasets = [previous_dataset]
            is_generated = True
  
        data_loader_previous = iter(utils.get_data_loader(
            ConcatDataset(previous_datasets), batch_size,
            cuda=self._is_on_cuda(), collate_fn=collate_fn,
        )) if previous_datasets else None

        data_cycle = cycle(data_loader)
        prev_data_cycle = cycle(data_loader_previous) if data_loader_previous is not None else None

        # define a tqdm progress bar.
        batch_count = len(data_loader)
        total_iteration = epochs*batch_count
        progress = tqdm(range(1, total_iteration+1))

        for iteration in progress:
            # decide from where to sample the training data.
            from_scholar = scholar is not None
            from_previous_datasets = bool(previous_datasets)
            cuda = self._is_on_cuda()

            # sample the real training data.
            x, y = next(data_cycle)
            x = Variable(x).cuda() if cuda else Variable(x)
            y = Variable(y).cuda() if cuda else Variable(y)

            # sample the replayed training data.
            if from_previous_datasets and not is_generated:
                x_, y_ = next(prev_data_cycle)
            elif from_scholar:
                x_, y_ = next(prev_data_cycle)
                # x_, y_ = scholar.sample(batch_size)
            else:
                x_ = y_ = None

            if x_ is not None and y_ is not None:
                x_ = Variable(x_).cuda() if cuda else Variable(x_)
                y_ = Variable(y_).cuda() if cuda else Variable(y_)

            # train the model with a batch.
            result = trainable.train_a_batch(
                x, y, x_=x_, y_=y_,
                importance_of_new_task=importance_of_new_task
            )

            # fire the callbacks on each iteration.
            for callback in (training_callbacks or []):
                epoch = iteration // batch_count
                callback(trainable, progress, epoch, iteration, total_iteration, result)


    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
