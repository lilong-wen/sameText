import cv2
import torch
from dataloader.datasets import SynthText
from dataloader.dataset_wrapper import DataSetWrapper

from utils import test_dataset_class
from utils import train_test_split


if __name__ == "__main__":

    train_folder = "/home/zju/w4/datasets/SynthText"

    # test dataset class
    # data_set = SynthText(train_folder)
    #
    # train_set, test_set = train_test_split(data_set, 0.8)
    #
    # print(data_set[0])
    # print(len(train_set))
    # print(len(test_set))

    # test dataset wrapper
    dataset = DataSetWrapper(batch_size=4,
                             num_workers=1,
                             valid_size=0.3,
                             input_shape="(224, 224, 3)",
                             s=1,
                             root_dir=train_folder)
    train_data, val_data = dataset.get_data_loaders()
    for batch_ndx, sample in enumerate(train_data):
        print(batch_ndx)
        print(sample[1])
        print(sample[0].shape)
