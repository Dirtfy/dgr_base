import os

import copy
import math

from collections import defaultdict


import torch
from torchvision import datasets, transforms
from PIL import ImageOps, Image
from torch.utils.data import ConcatDataset, Subset, Dataset

from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

from tqdm import tqdm


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    return image.view(c, h, w)


def _colorize_grayscale_image(image):
    return ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))


def get_dataset(name, train=True, permutation=None):
    dataset = (TRAIN_DATASETS[name] if train else TEST_DATASETS[name])()
    dataset.transform = transforms.Compose([
        dataset.transform,
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    return dataset

def files_in_directory(directory_path):
    # 폴더 내 모든 파일과 디렉토리 가져오기
    all_items = os.listdir(directory_path)
    # 파일만 필터링
    files = [f for f in all_items if os.path.isfile(os.path.join(directory_path, f))]
    return files

def tensorToPIL(tensor: torch.Tensor) -> Image:
    tensor = tensor.clone()
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # 정규화

    return ToPILImage()(tensor)

def save_as_image(tensor, file_path):
    image = tensorToPIL(tensor)
    image.save(file_path)
        
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = files_in_directory(folder_path)

        self.transform = ToTensor()

    def __len__(self):
         return len(self.image_paths)
    
    def get_label_from_image_path(self, image_path: str):
        name, extensoin = os.path.splitext(os.path.basename(image_path))
        return int(name.split('_')[-1])
    
    def __getitem__(self, index):
        # 이미지 로드
        img_path = os.path.join(self.folder_path, self.image_paths[index])
        image = self.transform(Image.open(img_path))
        label = torch.tensor([self.get_label_from_image_path(img_path)], dtype=torch.long)
        
        return image, label
    
def split_by_label(dataset):
        """
        주어진 클래스(target_class)에 해당하는 데이터만 포함된 하위 데이터셋 반환
        """

        # 라벨별 인덱스 수집
        label_to_indices = defaultdict(list)
        for i, (_, label) in tqdm(enumerate(dataset), desc="Splitting Dataset by label"):
            label_to_indices[label].append(i)

        # Subset 생성
        splited_dataset = {
            label: Subset(dataset, indices) 
            for label, indices in label_to_indices.items()
        }

        return splited_dataset


_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_MNIST_COLORIZED_TRAIN_TRANSFORMS = _MNIST_COLORIZED_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_CIFAR_TRAIN_TRANSFORMS = _CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]

_SVHN_TRAIN_TRANSFORMS = _SVHN_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]
_SVHN_TARGET_TRANSFORMS = [
    transforms.Lambda(lambda y: y % 10)
]


TRAIN_DATASETS = {
    'mnist': lambda: datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'mnist-color': lambda: datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_COLORIZED_TRAIN_TRANSFORMS)
    ),
    'cifar10': lambda: datasets.CIFAR10(
        './datasets/cifar10', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'cifar100': lambda: datasets.CIFAR100(
        './datasets/cifar100', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'svhn': lambda: datasets.SVHN(
        './datasets/svhn', split='train', download=True,
        transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS),
        target_transform=transforms.Compose(_SVHN_TARGET_TRANSFORMS),
    ),
    
    'mnist-0': lambda: torch.load("./datasets/mnist/MNIST/byLabel/0-train"),
    'mnist-1': lambda: torch.load("./datasets/mnist/MNIST/byLabel/1-train"),
    'mnist-2': lambda: torch.load("./datasets/mnist/MNIST/byLabel/2-train"),
    'mnist-3': lambda: torch.load("./datasets/mnist/MNIST/byLabel/3-train"),
    'mnist-4': lambda: torch.load("./datasets/mnist/MNIST/byLabel/4-train"),
    'mnist-5': lambda: torch.load("./datasets/mnist/MNIST/byLabel/5-train"),
    'mnist-6': lambda: torch.load("./datasets/mnist/MNIST/byLabel/6-train"),
    'mnist-7': lambda: torch.load("./datasets/mnist/MNIST/byLabel/7-train"),
    'mnist-8': lambda: torch.load("./datasets/mnist/MNIST/byLabel/8-train"),
    'mnist-9': lambda: torch.load("./datasets/mnist/MNIST/byLabel/9-train"),
}


TEST_DATASETS = {
    'mnist': lambda: datasets.MNIST(
        './datasets/mnist', train=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'mnist-color': lambda: datasets.MNIST(
        './datasets/mnist', train=False, download=True,
        transform=transforms.Compose(_MNIST_COLORIZED_TEST_TRANSFORMS)
    ),
    'cifar10': lambda: datasets.CIFAR10(
        './datasets/cifar10', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'cifar100': lambda: datasets.CIFAR100(
        './datasets/cifar100', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'svhn': lambda: datasets.SVHN(
        './datasets/svhn', split='test', download=True,
        transform=transforms.Compose(_SVHN_TEST_TRANSFORMS),
        target_transform=transforms.Compose(_SVHN_TARGET_TRANSFORMS),
    ),

    'mnist-0': lambda: torch.load("./datasets/mnist/MNIST/byLabel/0-test"),
    'mnist-1': lambda: torch.load("./datasets/mnist/MNIST/byLabel/1-test"),
    'mnist-2': lambda: torch.load("./datasets/mnist/MNIST/byLabel/2-test"),
    'mnist-3': lambda: torch.load("./datasets/mnist/MNIST/byLabel/3-test"),
    'mnist-4': lambda: torch.load("./datasets/mnist/MNIST/byLabel/4-test"),
    'mnist-5': lambda: torch.load("./datasets/mnist/MNIST/byLabel/5-test"),
    'mnist-6': lambda: torch.load("./datasets/mnist/MNIST/byLabel/6-test"),
    'mnist-7': lambda: torch.load("./datasets/mnist/MNIST/byLabel/7-test"),
    'mnist-8': lambda: torch.load("./datasets/mnist/MNIST/byLabel/8-test"),
    'mnist-9': lambda: torch.load("./datasets/mnist/MNIST/byLabel/9-test"),
}


DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist-color': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},

    'mnist-0': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-1': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-2': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-3': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-4': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-5': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-6': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-7': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-8': {'size': 32, 'channels': 1, 'classes': 1},
    'mnist-9': {'size': 32, 'channels': 1, 'classes': 1}
}
