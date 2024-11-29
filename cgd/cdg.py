import random

import torch

import dgr

from .classifier import Classifier
from .classifier_guidance import ClassifierGuidedDiffusion

class CDG(dgr.Generator):
    def __init__(self, 
                 model: ClassifierGuidedDiffusion,
                 n_classes):
        super().__init__()

        self.device = next(model.parameters()).device
        self.model = model
        self.n_classes = n_classes
        
        self.optimizer = None
        self.classifier_optimizer = None

    def sample(self, size, y):
        return self.model.sample(
            batch_size=size,
            label=y)
    
    def classifier_noise_train_a_batch(self, x, y) -> torch.Tensor:
        rand_diffusion_step = self.model.sample_diffusion_step(batch_size=x.size(0))
        rand_noise = self.model.sample_noise(batch_size=x.size(0))
        noisy_image = self.model.perform_diffusion_process(
            ori_image=x,
            diffusion_step=rand_diffusion_step,
            rand_noise=rand_noise,
        )

        self.classifier_optimizer.zero_grad()

        loss = self.model.classifier.get_loss(noisy_image, rand_diffusion_step, y)

        loss.backward()
        self.classifier_optimizer.step()

        return loss


    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()

        noise_train_loss = self.classifier_noise_train_a_batch(x, y)

        self.optimizer.zero_grad()
        
        new_task_loss = self.model.get_unet_loss(x, y)
        if x_ is not None and y_ is not None:
            old_task_loss = self.model.get_unet_loss(x_, y_)

            loss = (
                importance_of_new_task * new_task_loss +
                (1-importance_of_new_task) * old_task_loss
            )
        else:
            loss = new_task_loss

        loss.backward()
        self.optimizer.step()

        return {'c_loss': noise_train_loss.item(), 'g_loss': loss.item()}
    
    
class Classifier(dgr.Solver):
    def __init__(self,
                 model: Classifier):
        super().__init__()

        self.model = model


    def forward(self, x):
        t = torch.zeros(size=(x.shape[0], 1), dtype=torch.long)
        return self.model(x, t, None)
