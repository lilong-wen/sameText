from scipy import spatial
import os
import yaml
from dataloader.dataset_wrapper import DataSetWrapper
from transformers import AutoTokenizer
import argparse
from models.model import ModelCLR
import torch
import cv2
import numpy as np
from apex import amp
import torchvision.transforms as transforms
import h5py

checkpoints_folder = "./runs/Mar29_10-40-19_ubuntu/checkpoints/"

# xls = tokenizer(list(xls), return_tensors="pt", padding=True, truncation=truncation)

def get_img():

    img_db_1 = h5py.File('./task1/synthText_1.h5')
    img_db_2 = h5py.File('./task1/synthText_2.h5')

    dsets_1 = sorted(img_db_1['data'].keys())
    dsets_2 = sorted(img_db_2['data'].keys())

    image_data_1 = {}
    image_data_2 = {}

    for img_name_1 in dsets_1:
        img = img_db_1['data'][img_name_1][...]
        image_data_1[img_name_1] = img

    for img_name_2 in dsets_2:
        img = img_db_2['data'][img_name_2][...]
        image_data_2[img_name_2] = img

    return image_data_1, image_data_2


def similar_estimate(vec1, vec2):

    result = 1 - spatial.distance.cosine(vec1, vec2)

def compute_acc(fea1, fea2):

    total_len = len(fea1.keys())
    ac = 0
    for name1, fea1 in fea1.items():
        smallest_name = ''
        smallest_dis = 10000
        for name2, fea2 in fea2.items():
            dis = similar_estimate(fea1, fea2)
            if dis < smallest_dis:
                smallest_name = name2
                smallest_dis = dis

        if smallest_name == name1:
            ac += 1

    print(f"total num {total_len}, ac num {ac}")
    return ac/total_len


def get_simclr_pipeline_transform(input_img):
    h,w = input_img.shape[0:2]
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * 1,
                                          0.8 * 1,
                                          0.8 * 1,
                                          0.2 * 1)
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return data_transforms(input_img)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', type=str,
                       default='/home/zju/w4/datasets/SynthText')

    parser.add_argument('--task', default='img_retrieval',
                        choices=['img_retrieval',
                                 'txt_retrieval',
                                 'img_find_txt',
                                 'txt_find_img'])
    opt = parser.parse_args()

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    model = ModelCLR(**config["model"]).to("cuda")
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
    model.load_state_dict(state_dict)
    model.eval()

    model = amp.initialize(model,
                           opt_level='O2',
                           keep_batchnorm_fp32=True)


    tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])
    truncation = config['truncation']

    xls = 'abc'
    xls = tokenizer(list(xls),
                    return_tensors="pt",
                    padding=True,
                    truncation=truncation)

    xls.to('cuda')

    img_db_1, img_db_2 = get_img()

    feature_db_1 = {}
    feature_db_2 = {}

    for name, img in img_db_1.items():

        img  = get_simclr_pipeline_transform(cv2.resize(img,
                                                        (224, 224)))
        img = img.unsqueeze(0).to('cuda')

        feature, _ = model(img, xls)

        feature_db_1[name] = feature

    for name, img in img_db_2.items():

        img  = get_simclr_pipeline_transform(cv2.resize(img,
                                                        (224, 224)))
        img = img.unsqueeze(0).to('cuda')

        feature, _ = model(img, xls)

        feature_db_2[name] = feature

    value = compute_acc(feature_db_1, feature_db_2)
    # img = cv2.imread('./test.jpg')
    # img = cv2.resize(img, (224,224))
    # img = get_simclr_pipeline_transform(img)
    # img = img.unsqueeze(0).to('cuda')
    #
    #
    # img_emb, txt_emb = model(img, xls)
    #
    # print(img_emb.shape)
    # print(txt_emb.shape)
    #
    # print(img_emb)
    # print(8*"*")
    # print(txt_emb)
    print(value)
