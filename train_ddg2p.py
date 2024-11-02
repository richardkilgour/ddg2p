import argparse
import os
import sys

import torch
import yaml
from torch.profiler import profiler
from torch.utils.data import DataLoader

from src.data.utils import pad_collate, test_on_subset
from src.tools.G2pTrainer import G2pTrainer
from src.data.IpaDataset import IpaDataset
from src.model.ddg2pmodel import ddg2pModel

# torch.xpu is the API for Intel GPU support
if torch.xpu.is_available():
    # TODO Maybe this?
    print(f'{torch.xpu.is_available()=}')
    device = torch.device("xpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("CPU")
print(f'{device=}')

phase = 'all'

# You want slow code? Try this to make it worse, then fix it
profile = False


# Experiment: Architecture, languages, epochs etc

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Define your model based on config
    params = {'model': config['model']['model'], 'd_model': config['model']['d_model'],
              'n_layers': config['model']['n_layers']}
    model = ddg2pModel(params).to(device)

    # Define optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters())  # Otherwise defualts
    elif config['training']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Add training loop, etc.
    # Load the data: ['Language', 'Ortho', 'Pref', 'Phon']
    # Use dummy data unless we are in load data or all phase
    ipa_data = IpaDataset(config['data']['data_path'], config['data']['csv_name'],
                          languages=config['data']['languages'], dummy_data=phase not in ['load_data', 'all'],
                          max_length=124)

    # Split into train/test/valid
    train_subset, test_subset, valid_subset = torch.utils.data.random_split(ipa_data, [0.6, 0.2, 0.2],
                                                                            generator=torch.Generator().manual_seed(1))

    train_dataloader = DataLoader(train_subset, batch_size=config['data']['batch_size'], shuffle=True,
                                  collate_fn=pad_collate)

    # If the network exists, load it
    if os.path.isfile(config['model']['PATH']):
        model.load_state_dict(torch.load(config['model']['PATH'], weights_only=False))

    # TODO: Is this a good use case for a decorator?
    if profile:
        from torch._C._profiler import ProfilerActivity
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sort_by_keyword = "cuda_time_total"
        with profiler.profile(activities=activities, with_stack=False, profile_memory=True) as prof:
            trainer = G2pTrainer(model, train_dataloader, optimizer, device, config['model']['PATH'],
                                 test_subset=test_subset)
            trainer.train(config['model']['max_epochs'])
        print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=10))
    else:
        trainer = G2pTrainer(model, train_dataloader, optimizer, device, config['model']['PATH'],
                             test_subset=test_subset)
        trainer.train(config['training']['max_epochs'])

    print(f'testng on validation set...')
    correct_language, correct_phoneme, total_PER = test_on_subset(valid_subset, model, device)
    print(f'{correct_language=}\t{correct_phoneme=}\t{total_PER=}')


if __name__ == "__main__":
    main()
