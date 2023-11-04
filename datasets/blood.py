import os
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

class Blood(Dataset):

    def __init__(self, setname:str, args, transform=None):
        d = os.path.join(args.data_path, setname)
        dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]

        data = []
        label = []
        lb = -1

        for d in dirs:
            lb += 1
            for image_name in os.listdir(d):
                path = os.path.join(d, image_name)
                data.append(path)
                label.append(lb)

        self.data = data
        self.label = label

        mean = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean, std=std)

        self.image_size = 84

        if transform is None:
            if setname in ['train','trainval']:
                transforms_list = [
                    transforms.Resize((self.image_size+12,self.image_size+12), antialias=True),
                    transforms.RandomCrop(self.image_size, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            else:
                transforms_list = [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                ]
        else:
            self.transform = transform

        self.transform = transforms.Compose(
            transforms_list + [normalize]
        )

    def __len__(self):
        return len(self.data)
\
    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, path
    
class SSLBlood(Dataset):

    def __init__(self, setname:str, args):
        d = os.path.join(args.data_path, setname)
        dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]

        data = []
        label = []
        lb = -1

        for d in dirs:
            lb += 1
            for image_name in os.listdir(d):
                path = os.path.join(d, image_name)
                data.append(path)
                label.append(lb)

        self.data = data
        self.label = label

        mean = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean, std=std)

        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.4, hue=0.1)
        
        self.augmentation_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(args.size, args.size)[-2:],
                                         scale=(0.5,1.0)),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])

        self.identity_transform = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            normalize
        ])

        self.shared_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        img = Image.open(path).convert('RGB')
        img = self.shared_transform(img)
        image = []
        for _ in range(1):
            image.append(self.identity_transform(img).unsqueeze(0))
        for i in range(3):
            image.append(self.augmentation_transform(img).unsqueeze(0))
        return dict(data=torch.cat(image)), label



