from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import os
from torchvision.io import read_video
import numpy as np

class EgoExo(Dataset):
    def __init__(self, egoexo_dataset, data_directory):
        self.labels_list = [0] * len(os.listdir(data_directory))
        self.videos_list = os.listdir(data_directory)
        for i in range(len(os.listdir(data_directory))):
            video_path = os.path.join(data_directory, self.videos_list[i])
            label = video_path.split("\\")[-1].split(".")[0][-3:]
            if label == "EGO":
                self.labels_list[i] = 0
            else:
                self.labels_list[i] = 1

        self.egoexo_dataset = egoexo_dataset
        self.data_directory = data_directory
        self.input_transform = T.Compose([
            T.ToTensor(),
            T.Resize(size = (224, 224)),
            T.RandomHorizontalFlip(0.25),
            T.RandomVerticalFlip(0.25),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __len__(self) -> int:
        return len(self.egoexo_dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path = os.path.join(self.data_directory, self.egoexo_dataset[index])
        frames, _, _ = read_video(video_path)
        frames = frames.permute(0, 3, 1, 2)
        indices = np.sort(np.random.random_integers(0, len(frames)-1, 10))
        frames_list = []
        for i in indices:
            x = T.ToPILImage()(frames[i].squeeze_(0))
            frames_list.append(self.input_transform(x))
        
        out_tensor = torch.stack(frames_list, dim=0)
        out_label = torch.tensor(self.labels_list[index], dtype=torch.float32)
        return (out_tensor, out_label)