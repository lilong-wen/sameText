import argparse
from train import SimCLR
import yaml
from dataloader.dataset_wrapper import DataSetWrapper as \
    DataSetWrapper_origin
from dataloader_retrieval.dataset_wrapper import DataSetWrapper as \
    DataSetWrapper_retrieval


def main(params):

    if params.task =='retrieval':
        config = yaml.load(open("config_retrieval.yaml", "r"), Loader=yaml.FullLoader)
        dataset = DataSetWrapper_retrieval(config['batch_size'], **config['dataset'])
    else:
        config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        dataset = DataSetWrapper_origin(config['batch_size'], **config['dataset'])


    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='retrieval')
    args = parser.parse_args()

    main(args)
