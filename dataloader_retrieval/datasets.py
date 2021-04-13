import h5py
import math
import os
import random
import re

import cv2
import numpy as np
import scipy.io
import torch
import torch.utils.data
import torchvision


class SynthText_retrieval(torch.utils.data.Dataset):
    def __init__(self, root_1, root_2, transform=None):
        self.transform = transform
        self.root_1 = root_1
        self.root_2 = root_2

        self.db_1 = h5py.File(self.root_1, 'r')
        self.db_2 = h5py.File(self.root_2, 'r')

        assert(len(sorted(self.db_2['data'].keys())) == len(sorted(self.db_1['data'].keys())))
        self.length = len(sorted(self.db_1['data'].keys()))
        #sample_path = labels['imnames'][0, 1][0]
        #sample_boxes = np.transpose(labels['wordBB'][0, 1], (2, 1, 0))

        self.transform = transform

        self.normalizer = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        dsets_1 = sorted(self.db_1['data'].keys())
        dsets_2 = sorted(self.db_2['data'].keys())

        img_1 = self.db_1['data'][dsets_1[idx]][...]
        img_1_txt = self.db_1['data'][dsets_1[idx]].attrs['txt']
        img_2_positive = self.db_2['data'][dsets_2[idx]][...]
        img_2_positive_txt = self.db_2['data'][dsets_2[idx]].attrs['txt']
        random_num = random.random()
        if random_num > 0.5:
            random_idx = random.randint(0, self.length)
            if random_idx != idx:
                img_3_negtive = self.db_1['data'][dsets_1[random_idx]][...]
                img_3_negtive_txt = self.db_1['data'][dsets_1[random_idx]].attrs['txt']
            else:
                img_3_negtive = self.db_1['data'][dsets_1[idx + 1]][...]
                img_3_negtive_txt = self.db_1['data'][dsets_1[idx + 1]].attrs['txt']
        else:
            random_idx = random.randint(0, self.length)
            if random_idx != idx:
                img_3_negtive = self.db_2['data'][dsets_2[random_idx]][...]
                img_3_negtive_txt = self.db_2['data'][dsets_2[random_idx]].attrs['txt']
            else:
                img_3_negtive = self.db_2['data'][dsets_2[idx + 1]][...]
                img_3_negtive_txt = self.db_2['data'][dsets_2[idx + 1]].attrs['txt']

        sample = {'anchor': [img_1, img_1_txt.tolist()],
                  'positive': [img_2_positive, img_2_positive_txt.tolist()],
                  'negtive': [img_3_negtive, img_3_negtive_txt.tolist()]}

        if self.transform != None:
            sample = self.transform(sample)
        return sample
