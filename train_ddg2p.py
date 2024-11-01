import os

import torch
from torch._C._profiler import ProfilerActivity
from torch.profiler import profiler
from torch.utils.data import DataLoader

#from src.config.config import data_config
from experiments.experiment_1.config import data_config, model_config

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


# Load the data: ['Language', 'Ortho', 'Pref', 'Phon']
# Use dummy data unless we are in load data or all phase
ipa_data = IpaDataset(data_config['data_path'], data_config['csv_name'],
                      languages=data_config['languages'], dummy_data=phase not in ['load_data', 'all'],
                      max_length=124)

# Split into train/test/valid
train_subset, test_subset, valid_subset = torch.utils.data.random_split(ipa_data, [0.6, 0.2, 0.2],
                                                                        generator=torch.Generator().manual_seed(1))

train_dataloader = DataLoader(train_subset, batch_size=data_config['batch_size'], shuffle=True, collate_fn=pad_collate)

params = {'model': model_config['model'], 'd_model': model_config['d_model'], 'n_layers': model_config['n_layers']}
net = ddg2pModel(params).to(device)

# If the network exists, load it
if os.path.isfile(model_config['PATH']):
    net.load_state_dict(torch.load(model_config['PATH'], weights_only=False))

if phase in ['train', 'all']:
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # TODO: Is this a good use case for a decorator?
    if profile:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sort_by_keyword = "cuda_time_total"
        with profiler.profile(activities=activities, with_stack=False, profile_memory=True) as prof:
            trainer = G2pTrainer(net, train_dataloader, optimizer, device, model_config['PATH'], test_subset=test_subset)
            trainer.train(model_config['max_epochs'])
        print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=10))
    else:
        trainer = G2pTrainer(net, train_dataloader, optimizer, device, model_config['PATH'], test_subset=test_subset)
        trainer.train(model_config['max_epochs'])

print(f'testng on validation set...')
correct_language, correct_phoneme, total_PER = test_on_subset(valid_subset, net, device)
print(f'{correct_language=}\t{correct_phoneme=}\t{total_PER=}')
