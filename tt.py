import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms


from tqdm import tqdm
import time

import utils

import data

v = datasets.CIFAR10("/home/mskim/project/pytorch-deep-generative-replay/datasets/cifar10")
v = data.split_by_label(v)
# print(type(v[0][0][0]))
for i in range(10):
    v[i][3][0].save(f"/home/mskim/project/pytorch-deep-generative-replay/tt_{i}.png")
