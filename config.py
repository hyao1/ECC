from networks import resnet, densenet, inception
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from autoaugment import AutoAugImageNetPolicy
from torch.utils import data
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

class Config:
    def __init__(self, arg):
        self.lr = arg.lr
        self.gpu_id = arg.gpu_id
        self.dim = 2048
        self.dataset_name = arg.dataset_name
        self.model_path = arg.model_path
        self.batch_size = arg.bs
        self.img_size = arg.img_size
        self.epochs = arg.epochs
        self.loss = arg.loss
        self.lmd_1 = arg.lmd_1
        self.lmd_2 = arg.lmd_2

        self.step_distance = arg.psd
        self.time = arg.time
        self.text = arg.text
        self.loss = arg.loss

        self.model_name = arg.model
        self.train_dataloader, self.val_dataloader, self.num_class = self.set_dataloader()
        self.model = self.set_model()

    def set_model(self):
        if self.model_name == 'resnet50':
            net = resnet.resnet50(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features, self.num_class)

        elif self.model_name == 'resnet101':
            net = resnet.resnet101(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'resnet152':
            net = resnet.resnet152(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'densenet121':
            net = densenet.densenet121(pretrained=True)
            feature_num = net.classifier.in_features
            net.classifier = nn.Linear(feature_num, self.num_class)
            self.dim = 1024
        elif self.model_name == 'densenet161':
            net = densenet.densenet161(pretrained=True)
            feature_num = net.classifier.in_features
            net.classifier = nn.Linear(feature_num, self.num_class)

        elif self.model_name == 'inceptionv3':
            net = inception.inception_v3(pretrained=True)
            feature_num = net.fc.in_features
            net.fc = nn.Linear(feature_num, self.num_class)

        return net

    def set_dataloader(self):
        if self.dataset_name in ['cubbirds']:
            data_transform = {
                "train": transforms.Compose(
                    [transforms.Resize((600, 600)),
                     transforms.RandomCrop((self.img_size, self.img_size)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                ),
                "val": transforms.Compose(
                    [transforms.Resize((600, 600)),
                     transforms.CenterCrop((self.img_size, self.img_size)),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                )}
        elif self.dataset_name in ['inat']:
            data_transform = {
                'train': transforms.Compose([
                    transforms.Resize((400, 400), Image.BILINEAR),
                    transforms.RandomCrop((304, 304)),
                    transforms.RandomHorizontalFlip(),
                    AutoAugImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize((400, 400), Image.BILINEAR),
                    transforms.CenterCrop((304, 304)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])}
        elif self.dataset_name in ['stcars']:
            data_transform = {
                'train': transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])}
        elif self.dataset_name in ['aircrafts', 'nabirds']:
            data_transform = {
                "train": transforms.Compose(
                    [transforms.Resize((600, 600)),
                     transforms.RandomCrop((self.img_size, self.img_size)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                ),

                "val": transforms.Compose(
                    [transforms.Resize((600, 600)),
                     transforms.CenterCrop((self.img_size, self.img_size)),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                )
            }
        else:
            raise ValueError("no this dataset")
        
        if self.dataset_name == 'cubbirds':
            dataset_path_train = 'D:/datasets/cubbirds/trainval'
            dataset_path_val = 'D:/datasets/cubbirds/test'
            num_class = 200

        elif self.dataset_name == 'nabirds':
            dataset_path_train = '/home/20181214363/datasets/nabirds/train'
            dataset_path_val = '/home/20181214363/datasets/nabirds/val'
            num_class = 555

        elif self.dataset_name == 'stcars':
            dataset_path_train = '/home/20181214363/datasets/stcars/train'
            dataset_path_val = '/home/20181214363/datasets/stcars/val'
            num_class = 196

        elif self.dataset_name == 'aircrafts':
            dataset_path_train = '/home/20181214363/datasets/aircrafts/train'
            dataset_path_val = '/home/20181214363/datasets/aircrafts/val'
            num_class = 100

        elif self.dataset_name == 'inat':
            dataset_path_train = '/home/20181214363/datasets/inat'
            dataset_path_val = '/home/20181214363/datasets/inat'
            num_class = 5089
        else:
            raise ValueError("no this dataset")

        train_dataset = datasets.ImageFolder(root=dataset_path_train, transform=data_transform["train"])
        validate_dataset = datasets.ImageFolder(root=dataset_path_val, transform=data_transform["val"])

        if self.dataset_name == 'inat':
            train_dataset = INat2017(root=dataset_path_train, split='train', transform=data_transform["train"])
            validate_dataset = INat2017(root=dataset_path_val, split='val', transform=data_transform["val"])

        train_loader = data.DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=8)
        val_loader = data.DataLoader(validate_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=8)
        return train_loader, val_loader, num_class

    def print_info(self):
        info = '+---------------------------------------------------+\n' \
               '|| init lr : {:<10f}|| batch size   : {:<10d}||\n' \
               '|| img size: {:<10d}|| epochs       : {:<10d}||\n' \
               '|| parts   : {:<10d}|| step distance: {:<10d}||\n' \
               '|| model   : {:<10s}|| loss         : {:<10s}||\n' \
               '|| dataset : {:<10s}|| gpu id       : {:<10s}||\n' \
               '|| lmd_1   : {:<10f}|| lmd_2        : {:<10f}||\n' \
               '|| data augment: {:<33s}||\n' \
               '+--------------------------------------------------+'.format(self.lr, self.batch_size, self.img_size,
                                                                             self.epochs, 1,
                                                                             self.step_distance,
                                                                             self.model_name,
                                                                             self.loss,
                                                                             self.dataset_name,
                                                                             self.gpu_id,
                                                                             self.lmd_1, self.lmd_2,
                                                                             'RandomCrop, RandomHorizontalFlip')
        print(info)

class INat2017(Dataset):
    """`iNaturalist 2017 <https://github.com/visipedia/inat_comp/blob/master/2017/README.md>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'train_val_images/'
    file_list = {
        'imgs': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val_images.tar.gz',
                 'train_val_images.tar.gz',
                 '7c784ea5e424efaec655bd392f87301f'),
        'annos': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val2017.zip',
                  'train_val2017.zip',
                  '444c835f6459867ad69fcb36478786e7')
    }

    def __init__(self, root, split='train', transform=None):
        super(INat2017, self).__init__(root, transform=transform)
        self.loader = default_loader

        anno_filename = split + '2017.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']

    def __getitem__(self, index):
        path = os.path.join(self.root, self.images[index]['file_name'])
        target = self.annos[index]['category_id']

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images)