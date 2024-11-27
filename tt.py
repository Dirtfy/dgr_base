import torch

from tqdm import tqdm
import time

a = torch.tensor([[0, 0],[0, 0]])
print(a.shape)
a = a[:, :, None, None]
print(a.shape)
