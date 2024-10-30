import os

import torch
from torch._C._profiler import ProfilerActivity
from torch.profiler import profiler
from torch.utils.data import DataLoader

#from src.config.config import data_config
from experiments.experiment_1.config import data_config, model_config

from src.data.utils import pad_collate, BOS, SEP, EOS, calculate_per
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
            trainer = G2pTrainer(net, train_dataloader, optimizer, device, 10, model_config['PATH'])
            trainer.train(model_config['max_epochs'])
        print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=10))
    else:
        trainer = G2pTrainer(net, train_dataloader, optimizer, device, 10, model_config['PATH'])
        trainer.train(model_config['max_epochs'])


# Test it

def get_metrics(w):
    out = net.generate(w, device)
    try:
        EOS_pos = out.index(EOS)
        SEP_pos = out.index(SEP)
        # TODO: Splitting on bytes is a pain
        ortho = out[1:EOS_pos]
        lan = out[EOS_pos + 1:SEP_pos]
        phon = out[SEP_pos + 1:-1]

        # TODO: Was the language correct?
        targ_lan = valid_subset.dataset.data.iloc[i]['Language']
        correct_language = targ_lan == lan
        # TODO: Was the phoneme sequence correct? WER
        targ_phn = valid_subset.dataset.data.iloc[i]['Phon']
        correct_phoneme = targ_phn == phon
        # TODO: Were the phonemes correct? PER
        PER = calculate_per(phon, targ_phn)
        if i % 100 == 0:
            print(f'{ortho=}\t{lan=}\t{phon=}\t{targ_lan=}\t{targ_phn=}\t{PER=}')
        return correct_language, correct_phoneme, PER
    except:
        # Early networks fail this. One epoch seems enough to get vaild UTF-8
        if i % 100 == 0:
            print(f'PARSE FAILED: {out}')
    return False, False, 0


correct_language = 0
correct_phoneme = 0
total_PER = 0

for i in valid_subset.indices:
    # TODO: I made this real ugly for some reason
    correct_language_, correct_phoneme_, total_PER_ = get_metrics(valid_subset.dataset.data.iloc[i]['Ortho'])
    correct_language += correct_language_
    correct_phoneme += correct_phoneme_
    total_PER += total_PER_
print(f'{correct_language=}\t{correct_phoneme=}\t{total_PER=}')
