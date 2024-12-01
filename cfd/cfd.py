import random

import torch

import dgr

from .diffusion import GaussianDiffusion
from .embedding import ConditionalEmbedding

class CFD(dgr.Generator):
    def __init__(self,
                 model: GaussianDiffusion,
                 embeder: ConditionalEmbedding,
                 img_shape,
                 n_classes):
        super().__init__()

        self.model = model
        self.embeder = embeder

        self.img_shape = img_shape
        self.n_classes = n_classes

        self.optimizer = None

    def sample(self, size, y):        
        sample_shape = (size, *self.img_shape)

        model_kwargs = {}
        model_kwargs['cemb'] = self.embeder(y)

        return self.model.sample(shape=sample_shape, **model_kwargs)

    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()

        self.optimizer.zero_grad()

        model_kwargs = {}
        model_kwargs['cemb'] = self.embeder(y)

        new_task_loss = self.model.trainloss(x, **model_kwargs)
        if x_ is not None and y_ is not None:
            _model_kwargs = {}
            _model_kwargs['cemb'] = self.embeder(y)

            old_task_loss = self.model.trainloss(x_, **_model_kwargs)

            loss = (
                importance_of_new_task * new_task_loss +
                (1-importance_of_new_task) * old_task_loss
            )
        else:
            loss = new_task_loss

        loss.backward()
        self.optimizer.step()

        return {'c_loss': 0.0, 'g_loss': loss.item()}