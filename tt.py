import os
import random

from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

def tensorToPIL(tensor: torch.Tensor) -> Image:
    tensor = tensor.clone()
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # 정규화

    return ToPILImage()(tensor)

def save_as_image(tensor, file_path):
    image = tensorToPIL(tensor)
    image.save(file_path)

def files_in_directory(directory_path):
    # 폴더 내 모든 파일과 디렉토리 가져오기
    all_items = os.listdir(directory_path)
    # 파일만 필터링
    files = [f for f in all_items if os.path.isfile(os.path.join(directory_path, f))]
    return files

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
  

def generate_dataset(folder_path, model_path, total_size, batch_size, label_pool, device) -> Dataset:

    model = torch.load(model_path).to(device)

    file_index = 0

    batch_schedule = [batch_size] * (total_size//batch_size) + [total_size%batch_size]

    # 균등하게 라벨 scheduling
    label_schedule = []
    while len(label_schedule) < total_size:
        for label in label_pool:
            label_schedule.append(label)

            if len(label_schedule) == total_size:
                break
    
    random.shuffle(label_schedule)

    # batch schedule에 맞게 label schedule 자르기
    label_batches = []
    start = 0
    for batch in batch_schedule:
        label_batches.append(label_schedule[start:start + batch])
        start += batch

    # 샘플링 - {파일 번호}_{라벨}.png 로 folder path에 저장
    for batch, labels in zip(batch_schedule, label_batches):
        labels = torch.tensor(labels).to(device)
            
        images = model.sample(batch, labels)

        for image, label in zip(images, labels):
            file_name = f"{file_index}_{label}.png"
            save_as_image(image, os.path.join(folder_path, file_name))

            file_index += 1

    return ImageFolderDataset(folder_path=folder_path)
    
