import torch as t
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision as tv
import numpy as np

from config import opt

def extract_data(mode):
    
    transforms = tv.transforms.Compose([tv.transforms.RandomRotation(30),
                                       tv.transforms.RandomResizedCrop(224),
                                       tv.transforms.RandomVerticalFlip(),
                                       tv.transforms.ToTensor(),
                                       tv.transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)

    indices = np.random.permutation(len(dataset))
    train_split = int(np.floor(len(dataset)*opt.train_split))
    val_split = int(np.floor(len(dataset)*(opt.train_split+opt.val_split)))
    
    train_sampler = SubsetRandomSampler(indices[:train_split])
    val_sampler = SubsetRandomSampler(indices[train_split:val_split])
    test_sampler = SubsetRandomSampler(indices[val_split:])
    
    train_loader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         sampler=train_sampler
                                         )
    val_loader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         sampler=val_sampler
                                         )
    test_loader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         sampler=test_sampler
                                         )
    
    if mode == "train":
        return train_loader
    elif mode == "val":
        return val_loader
    elif mode == "test":
        return test_loader
