import argparse
import os
import logging

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.utils import test_on_subset, BucketBatchSampler, pad_collate
from src.tools.G2pTrainer import G2pTrainer
from src.data.IpaDataset import IpaDataset
from src.model.G2pModel import G2pModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    # torch.xpu is the API for Intel GPU support
    device = torch.device("xpu")
else:
    device = torch.device("CPU")
logger.info(f'Device used: {device}')

# You want slow code? Try this to make it worse, then fix it
profile = False


def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def profile_func(func):
    def wrapper(*args):
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sort_by_keyword = "cuda_time_total"
        with profiler.profile(activities=activities, with_stack=False, profile_memory=True) as prof:
            retval = func(args)
            logger.info(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=10))
        return retval

    if profile:
        return wrapper
    else:
        return func


def main():
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    model_params = {'model': config['model']['model'], 'd_model': config['model']['d_model'],
                    'n_layers': config['model']['n_layers'], 'expand_factor': config['model']['expand_factor']}
    model = G2pModel(model_params).to(device)

    # Define optimizer
    # TODO: Get params from config
    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters())  # Otherwise defaults
    else:  # config['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Add training loop, etc.
    # Load the data: ['Language', 'Ortho', 'Pref', 'Phon']
    remove_spaces = False
    if 'remove_spaces' in config['data']:
        remove_spaces = config['data']['remove_spaces']

    ipa_data = IpaDataset(config['data']['data_path'], config['data']['csv_name'],
                          languages=config['data']['languages'], max_length=124, remove_spaces=remove_spaces)

    # Split into train/test/valid subsets
    train_split = config['data'].get('train_split', 0.6)
    test_split = config['data'].get('test_split', 0.2)
    validation_split = 1. - train_split - test_split
    train_subset, test_subset, valid_subset = torch.utils.data.random_split(ipa_data,
                                                                            [train_split, test_split, validation_split],
                                                                            generator=torch.Generator().manual_seed(1))

    bucket_sampler = BucketBatchSampler(train_subset, batch_size=config['training']['batch_size'])

    # If the network exists, load it
    if os.path.isfile(config['model']['PATH']):
        model.load_state_dict(torch.load(config['model']['PATH']))

    @profile_func
    def train_it():
        train_dataloader = DataLoader(train_subset, batch_sampler=bucket_sampler, collate_fn=pad_collate,
                                      pin_memory=True)
        trainer = G2pTrainer(model, train_dataloader, optimizer, device, config['model']['PATH'],
                             test_subset=test_subset)
        trainer.train(config['training']['max_epochs'])

    train_it()

    logger.info(f'testng on validation set...')
    correct_language, total_wer, total_per = test_on_subset(valid_subset, model, device)
    logger.info(f'{correct_language=}\t{total_wer=}\t{total_per=}')


if __name__ == "__main__":
    main()
