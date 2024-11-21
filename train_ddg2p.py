import argparse
import logging
import os

import torch
import yaml
from numpy import Inf
from torch.utils.data import DataLoader

from src.data.DataConstants import device, PROFILING
from src.data.DataUtils import test_on_subset, pad_collate
from src.data.BucketBatchSampler import BucketBatchSampler
from src.tools.G2pTrainer import G2pTrainer
from src.data.IpaDataset import IpaDataset, IpaDatasetCollection
from src.model.G2pModel import G2pModel

logger = logging.getLogger(__name__)


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
        with profiler.PROFILING(activities=activities, with_stack=False, profile_memory=True) as prof:
            retval = func(args)
            logger.info(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=10))
        return retval

    if PROFILING:
        return wrapper
    else:
        return func


def restore_cp(cp_file, model, optimizer):
    # Load the checkpoint
    checkpoint = torch.load(cp_file)
    # Load the model state dictionary
    model.load_state_dict(checkpoint['model_state'])
    # Load the optimizer state dictionary
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    training_metrics = {
        'epoch': checkpoint['epoch'],
        'LER': checkpoint['LER'],
        'WER': checkpoint['WER'],
        'PER': checkpoint['PER'],
        'train_loss': checkpoint['train_loss'],
        'no_improvement_count': checkpoint['no_improvement_count'],
    }
    return training_metrics


def create_model(config):
    model_params = {'model': config['model']['model'], 'd_model': config['model']['d_model'],
                    'n_layers': config['model']['n_layers'], 'expand_factor': config['model']['expand_factor']}
    model = G2pModel(model_params).to(device)
    return model


def create_optimizer(model, config):
    # Define optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters())  # Otherwise defaults
    else:  # config['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer


def gather_data(config):
    # Load the data: ['Language', 'Ortho', 'Pref', 'Phon']
    remove_spaces = False
    if 'remove_spaces' in config['data']:
        remove_spaces = config['data']['remove_spaces']

    if isinstance(config['data']['data_name'], list):
        ipa_data = IpaDatasetCollection(config['data']['data_path'], config['data']['data_name'])
    else:
        # Split into train/test/valid subsets
        train_split = config['data'].get('train_split', 0.6)
        test_split = config['data'].get('test_split', 0.2)
        validation_split = 1. - train_split - test_split

        ipa_data = IpaDataset(config['data']['data_path'], config['data']['data_name'],
                              [train_split, test_split, validation_split],
                              languages=config['data']['languages'], max_length=124, remove_spaces=remove_spaces)
    return ipa_data


def main():
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    model = create_model(config)
    dataset = gather_data(config)

    optimizer = create_optimizer(model, config)

    # Set any default training metrics
    training_metrics = {
        'epoch': 0,
        'LER': 1.1,
        'WER': 1.1,
        'PER': 1.1,
        'train_loss': Inf,
        'no_improvement_count': 0,
    }

    # Restore any checkpoint that is found
    cp_file = config['model']['PATH'] + 'best_mamba_model.cp'
    if os.path.isfile(cp_file):
        training_metrics = restore_cp(cp_file, model, optimizer)
    else:
        cp_file = config['model']['PATH'] + 'latest_mamba_model.cp'
        if os.path.isfile(cp_file):
            training_metrics = restore_cp(cp_file, model, optimizer)

    logger.debug(f'Creating buckets...')
    bucket_sampler = BucketBatchSampler(dataset.train_subset, batch_size=config['training']['batch_size'])

    @profile_func
    def train_it():
        logger.debug(f'Training starts...')
        train_dataloader = DataLoader(dataset.train_subset, batch_sampler=bucket_sampler, collate_fn=pad_collate,
                                      pin_memory=True)
        trainer = G2pTrainer(model, train_dataloader, optimizer, device, config['model']['PATH'],
                             test_subset=dataset.test_subset)
        trainer.train(config['training']['max_epochs'], training_metrics=training_metrics)

    train_it()

    # Reload the best network
    cp_file = config['model']['PATH'] + 'best_mamba_model.cp'
    assert os.path.isfile(cp_file)
    restore_cp(cp_file, model, optimizer)

    logger.info(f'testng on validation set...')
    language_cm, total_wer, total_per = test_on_subset(dataset.valid_subset, model, beam_width=1)
    total_ler = 1. - language_cm.true_positive_rate()

    logger.info(f'{total_ler=:.5%}\t{total_wer=:.5%}\t{total_per=:.5%}')
    logger.info(f'{language_cm}')


if __name__ == "__main__":
    main()
