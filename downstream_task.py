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

checkpoints_folder = "./runs/Mar18_10-57-42_ubuntu/checkpoints/"

# xls = tokenizer(list(xls), return_tensors="pt", padding=True, truncation=truncation)

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


    img = cv2.imread('./test.jpg')
    img = cv2.resize(img, (224,224))
    img = get_simclr_pipeline_transform(img)
    img = img.unsqueeze(0).to('cuda')

    xls = 'abc'
    xls = tokenizer(list(xls),
                    return_tensors="pt",
                    padding=True,
                    truncation=truncation)

    xls.to('cuda')

    img_emb, txt_emb = model(img, xls)

    print(img_emb.shape)
    print(txt_emb.shape)

    print(img_emb)
    print(8*"*")
    print(txt_emb)
