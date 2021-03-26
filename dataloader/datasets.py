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


def point_dist_to_line(p1, p2, p3):
    """Compute the distance from p3 to p2-p1."""
    if not np.array_equal(p1, p2):
        return np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    else:
        return np.linalg.norm(p3 - p1)


IN_OUT_RATIO = 4
IN_SIDE = 640
OUT_SIDE = IN_SIDE // IN_OUT_RATIO


class ICDAR2015(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        self.img_dir = 'ch4_training_images'
        self.labels_dir = 'ch4_training_localization_transcription_gt'
        self.image_prefix = []
        self.pattern = re.compile('^' + '(\\d+),' * 8 + '(.+)$')
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
        for dirEntry in os.scandir(os.path.join(root, 'ch4_training_images')):
            self.image_prefix.append(dirEntry.name[:-4])

    def __len__(self):
        return len(self.image_prefix)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(os.path.join(self.root, self.img_dir), self.image_prefix[idx] + '.jpg'), cv2.IMREAD_COLOR).astype(np.float32)
        quads = []
        texts = []
        lines = [line.rstrip('\n') for line in open(os.path.join(os.path.join(self.root, self.labels_dir), 'gt_' + self.image_prefix[idx] + '.txt'),
                                                    encoding='utf-8-sig')]
        for line in lines:
            matches = self.pattern.findall(line)[0]
            numbers = np.array(matches[:8], dtype=float)
            quads.append(numbers.reshape((4, 2)))
            texts.append('###' != matches[8])
        return transform(img, np.stack(quads), texts, self.normalizer, self)


class SynthText(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.root = root
        self.labels = scipy.io.loadmat(os.path.join(root, 'gt.mat'))

        #sample_path = labels['imnames'][0, 1][0]
        #sample_boxes = np.transpose(labels['wordBB'][0, 1], (2, 1, 0))

        self.normalizer = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return self.labels['imnames'].shape[1]

    def __getitem__(self, idx):

        img = cv2.imread(os.path.join(self.root,
                                      self.labels['imnames'][0, idx][0]),
                         cv2.IMREAD_COLOR)

        coordinates = self.labels['wordBB'][0, idx]

        txt_str = ""
        for words in self.labels["txt"][0][idx]:
            txt_str += " " + " ".join([w.strip() for w in words.split("\n")])
        txt_str = txt_str.strip().split(" ")

        coord_text_list = []

        with_coord = False

        if len(coordinates.shape) > 2:
            for j in range(coordinates.shape[2]):
                x1 = int(coordinates[0][0][j])
                y1 = int(coordinates[1][0][j])
                x2 = int(coordinates[0][1][j])
                y2 = int(coordinates[1][1][j])
                x3 = int(coordinates[0][2][j])
                y3 = int(coordinates[1][2][j])
                x4 = int(coordinates[0][3][j])
                y4 = int(coordinates[1][3][j])

                # with coordinates
                # coord_text_list.append(([(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                #                         [txt_str[j]]))

                # without coordinates
                # coord_text_list.append(txt_str[j])

                # only coordinates
                coord_text_list.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

        else:
            x1 = int(coordinates[0][0])
            y1 = int(coordinates[1][0])
            x2 = int(coordinates[0][1])
            y2 = int(coordinates[1][1])
            x3 = int(coordinates[0][2])
            y3 = int(coordinates[1][2])
            x4 = int(coordinates[0][3])
            y4 = int(coordinates[1][3])

            # without coordinates
            # coord_text_list.append(txt_str[0])

            # with coordinates
            # coord_text_list.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)],txt_str[0])

            # only coordiante
            coord_text_list.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

        # return img, coord_text_list
        if not with_coord:
            sample = {'image': img, 'phrase': " ".join(txt_str)}
        else:
            sample = {'image': img, 'phrase': " ".join(txt_str), 'coordinates': coord_text_list}

        if self.transform:
            sample = self.transform(sample)

        return sample
