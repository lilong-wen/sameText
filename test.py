from transformers import AutoTokenizer
import os
import yaml
import cv2
import torch

from models.model import ModelCLR

checkpoints_folder = "./runs/Mar18_10-57-42_ubuntu/checkpoints/"

if __name__ == "__main__":

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    model = ModelCLR(**config["model"]).to("cuda")
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
    model.load_state_dict(state_dict)

    model.eval()

    img = cv2.imread("./test.jpg")
    img = cv2.resize(img, (224,224))
    print(img.shape)
    #tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])
    #truncation = config['truncation']

    # txt = "a b c"
    # xls = tokenizer(list(txt),
    #                 return_tensors="pt",
    #                 padding=True,
    #                 truncation=truncation)

    img = torch.from_numpy(img).float().to('cuda')
    img = img.unsqueeze(0)
    img = img.permute((0,3,1,2))
    #xls.to("cuda")

    xls = torch.tensor([1,1,1,1], dtype=torch.float).to("cuda")
    img_f, _ = model(img, xls)

    print(img_f)
