import os

import random

import torch

import dgr

from .classifier import Classifier
from .classifier_guidance import ClassifierGuidedDiffusion

from data import save_as_image

class Id2(dgr.Generator):
    def __init__(self, 
                 model: ClassifierGuidedDiffusion,
                 n_classes,
                 lambda_cg,
                 lambda_cfg,
                 cg_cfg_ratio):
        super().__init__()

        self.device = next(model.parameters()).device
        self.model = model
        self.n_classes = n_classes

        self.lambda_cg = lambda_cg
        self.lambda_cfg = lambda_cfg
        self.cg_cfg_ratio = cg_cfg_ratio

        self.label_pool = []
        
        self.optimizer = None
        self.classifier_optimizer = None


    def __sample(self, lambda_cg, lambda_cfg, y, batch_size):
        assert batch_size > 0

        # if len(self.label_pool) == 0:
        #     label = None
        # else:
        #     label = torch.tensor(
        #         random.choices(self.label_pool, k=batch_size)
        #         ).to(self.model.device)

        return self.model.sample(
            batch_size=batch_size,
            label=y,
            lambda_cg=lambda_cg,
            lambda_cfg=lambda_cfg)

    def sample(self, size, y=None):
        return self.__sample(self.lambda_cg, self.lambda_cfg, y, size)
        # cg_size = int(self.cg_cfg_ratio * size)
        # cfg_size = size - cg_size

        # mode_cg = self.__sample(
        #     lambda_cg=self.lambda_cg,
        #     lambda_cfg=self.lambda_cfg,
        #     batch_size=cg_size)
        
        # path = os.path.join('.', 'sample','test_2','cg')
        # os.makedirs(path, exist_ok=True)
        # for i, image in enumerate(mode_cg):
        #     save_as_image(image, os.path.join(path, f'{i}.png'))
        
        # mode_cfg = self.__sample(
        #     lambda_cg=self.lambda_cg,
        #     lambda_cfg=self.lambda_cfg,
        #     batch_size=cfg_size)
        
        # path = os.path.join('.', 'sample','test_2','cfg')
        # os.makedirs(path, exist_ok=True)
        # for i, image in enumerate(mode_cfg):
        #     save_as_image(image, os.path.join(path, f'{i}.png'))
        
        # concat = torch.concat([mode_cg, mode_cfg], dim=0)

        # return concat
    
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

        torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=1.0)

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

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {'c_loss': noise_train_loss.item(), 'g_loss': loss.item()}
    
    
class Classifier(dgr.Solver):
    def __init__(self,
                 model: Classifier):
        super().__init__()

        self.model = model


    def forward(self, x):
        t = torch.zeros(size=(x.shape[0],), dtype=torch.long)
        return self.model(x, t, None)
