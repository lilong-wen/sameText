import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from dataloader.gaussian_blur import GaussianBlur
from torchvision import datasets
from .datasets import SynthText_retrieval

np.random.seed(0)



class DataSetWrapper(object):
    def __init__(self,
                 batch_size,
                 num_workers,
                 valid_size,
                 input_shape,
                 s,
                 root_dir_1,
                 root_dir_2):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2


    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        train_dataset = SynthText_retrieval(self.root_dir_1, self.root_dir_2,
                                  transform=SimCLRDataTransform(data_augment)
                                    )

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        #                                transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s,
                                              0.8 * self.s,
                                              0.8 * self.s,
                                              0.2 * self.s)
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale((self.input_shape[0], self.input_shape[1])),
            transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            #   transforms.RandomApply([color_jitter], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            #   GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform_image):
        self.transform_image = transform_image

    def append_empty(self,length, list_):

        diff_len = length - len(list_)
        if diff_len < 0:
            raise AttributeError('Length error list is too long.')
        return list_ + [''] * diff_len

    def __call__(self, sample):

        xi_anchor = self.transform_image(sample['anchor'][0])
        xi_positive = self.transform_image(sample['positive'][0])
        xi_negtive = self.transform_image(sample['negtive'][0])
        # xl_anchor = [item.decode('utf-8') for item in sample['anchor'][1]]
        # xl_positive = [item.decode('utf-8') for item in sample['positive'][1]]
        # xl_negtive = [item.decode('utf-8') for item in sample['negtive'][1]]
        # xl_anchor = self.append_empty(50, list(sample['anchor'][1]))
        # xl_positive = self.append_empty(50, list(sample['positive'][1]))
        # xl_negtive = self.append_empty(50, list(sample['negtive'][1]))

        xl_anchor = ' '.join([item.decode('utf-8') for item in sample['anchor'][1]])
        xl_positive = ' '.join([item.decode('utf-8') for item in sample['positive'][1]])
        xl_negtive = ' '.join([item.decode('utf-8') for item in sample['negtive'][1]])


        # return [[xi_anchor, xi_positive, xi_negtive],\
        #        [xl_anchor, xl_positive, xl_negtive]]
        return xi_anchor, xi_positive, xi_negtive, \
            xl_anchor, xl_positive, xl_negtive
