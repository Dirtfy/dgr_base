import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms


from tqdm import tqdm
import time

import utils

v = datasets.CIFAR10("/home/mskim/project/pytorch-deep-generative-replay/datasets/cifar10")
v[0][0].save("/home/mskim/project/pytorch-deep-generative-replay/tt.png")
