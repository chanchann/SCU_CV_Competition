# -*- coding: UTF-8 -*-
import os
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import one_hot_encoding as ohe
import setting
from denoise_proc import get_clear_bin_image


class mydataset(Dataset):

    def __init__(self, folder, transform=None, denoise=False, channels=1):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform
        self.denoise = denoise
        self.channels = channels

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.denoise:
           image = get_clear_bin_image(image)
        if self.channels == 3:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = ohe.encode(image_name.split('_')[0]) 
        return image, label,image_root

transform = transforms.Compose([
    # transforms.ColorJitter(),
    #transforms.Grayscale(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def data_split(dataset, val_size=0.1, shuffle=True):
    #batch_size = 1
    random_seed= 42

# Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=128, #shuffle=shuffle,
                                           sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=1, #shuffle=shuffle,
                                                sampler=val_sampler, num_workers=4)
    return train_loader, val_loader


dataset = mydataset(setting.TRAIN_DATASET_PATH, 
transform=transform, 
denoise=False, 
channels=3)
train_loader, test_loader = data_split(dataset)

def get_train_data_loader():

    #dataset = mydataset(setting.TRAIN_DATASET_PATH, transform=transform)
    #train_loader, _ =  data_split(dataset)
    
    return train_loader

def get_test_data_loader():
    #dataset = mydataset(setting.TEST_DATASET_PATH, transform=transform)
    #_, test_loader = data_split(dataset)
    return test_loader

def get_predict_data_loader():
    dataset = mydataset(setting.PREDICT_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=False)

