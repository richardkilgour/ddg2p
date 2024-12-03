import csv
import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig

from src.data.IpaDataset import IpaDataset
from src.data.DataUtils import string_to_class, test_on_subset, tensor_to_utf8
from src.data.DataConstants import PAD, BOS, EOS, PROFILING, device

if PROFILING:
    from torch.autograd import profiler

logger = logging.getLogger(__name__)


def profile_func(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if PROFILING:
                with profiler.record_function(name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class G2pModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        d_model = params['d_model']
        self.embedding = nn.Embedding(256, d_model)

        n_layers = params['n_layers']
        expand_factor = params['expand_factor']
        self.model = params['model']
        if self.model == 'mamba':
            config = MambaConfig(d_model=d_model, n_layers=n_layers, expand_factor=expand_factor)
            self.recurrence = Mamba(config)
        elif self.model == 'gru':
            self.recurrence = nn.GRU(256, d_model, n_layers)
        # TODO: This used to be bias=False for some reason. Why did I do that???
        self.probs = nn.Linear(d_model, 256, bias=True)

    @profile_func('EMBEDDING')
    def _embed(self, inputs):
        return self.embedding(inputs)

    @profile_func('RECURRENCE')
    def _recurrence(self, inputs):
        return self.recurrence(inputs)

    @profile_func('LINEAR_OUT')
    def _linear(self, inputs):
        return self.probs(inputs)

    def forward(self, input_ids):
        x = self._embed(input_ids)
        x = self._recurrence(x)
        x = self._linear(x)
        return x

    def generate(self,
                 prompts: list[str],
                 max_tokens: int = 100,
                 ):
        self.eval()

        encoded_prompts = []
        for prompt in prompts:
            b_prompt = BOS + prompt + EOS
            b_encoded = string_to_class(b_prompt)
            encoded_prompts.append(torch.tensor(b_encoded, dtype=torch.long).to(device))

        # Any sequence that has terminated, we can ignore
        done_mask = [0] * len(encoded_prompts)
        for token_n in range(max_tokens):
            # Any short sequences should be sent through the network first,
            #  so that we can add to it until we have all the sequences in the batch with the same length
            shortest_item = min([t.size(0) for t in encoded_prompts])
            short_mask = [0 if x.size(0) > shortest_item else 1 for x in encoded_prompts]

            mask = [short and not done for short, done in zip(short_mask, done_mask)]

            filtered_tensors = [tensor for tensor, m in zip(encoded_prompts, mask) if m == 1]

            if not filtered_tensors:
                break

            with torch.no_grad():
                token_logits = self(torch.stack(filtered_tensors))
            next_token_logits = token_logits[:, -1]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_indices = torch.argmax(probs, dim=-1)[:, None]
            next_value = next_indices.tolist()

            update_counter = 0
            for i, (t, m) in enumerate(zip(encoded_prompts, mask)):
                if m == 1:
                    encoded_prompts[i] = torch.cat([t, next_indices[update_counter]])
                    if next_value[update_counter][0] == ord(EOS) or next_value[update_counter][0] == ord(PAD):
                        done_mask[i] = 1
                    update_counter += 1
                # Add PAD to sequences that have finished decoding to keep their length in sync
                if done_mask[i]:
                    encoded_prompts[i] = torch.cat([t, torch.as_tensor([ord(PAD)]).to(device)])

        return [tensor_to_utf8(ids.byte()) for ids in encoded_prompts]

    def _get_next_beam(self, sequences, beam_width):
        # List of candidate sequences, and their probabilities
        new_seq = []
        # For every sequence, run the network, and grab the k most probable continuations
        # TODO: This can be done in one batch
        for seq, prob in sequences:
            with torch.no_grad():
                next_token_logits = self(torch.unsqueeze(seq, 0))[:, -1]
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1).squeeze()
            # Add these to the current candidate list, taking the best k overall
            values, indices = torch.topk(probs, beam_width)
            for v, i in zip(values, indices):
                new_seq.append([torch.cat((seq, torch.unsqueeze(i, 0))), (prob * v).item()])
        return sorted(new_seq, key=lambda tup: -tup[1])[:beam_width]

    def generate_beam(self,
                      prompt: str,
                      beam_width=5,
                      max_tokens: int = 100,
                      ):
        """
        Return a list of the n-best predicted sequences, plus the probability of each
        [ tensor([id0, id1, id2...]), prob ]
        """
        self.eval()

        n_best = list()

        b_prompt = BOS + prompt + EOS
        b_encoded = string_to_class(b_prompt)

        # Convert byte data to a numpy array, batch size 1
        input_ids = [[torch.tensor(b_encoded, dtype=torch.long).to(device), 1.]]

        # Find a list of candidate sequences, and their probabilities
        for _ in range(max_tokens):
            input_ids = self._get_next_beam(input_ids, beam_width)
            # Scan the list backwards so we can safely remove items
            for i in range(len(input_ids) - 1, -1, -1):
                seq, prob = input_ids[i]
                # We can stop tracking this if we've already found a better candidate, as probability will not increase!
                # Or if the sequence has hit a termination character
                if (n_best and prob < n_best[0][1]) or seq[-1] == ord(EOS) or seq[-1] == ord(PAD):
                    # Remove it from the beam search
                    input_ids.pop(i)
                    # In the case of EOS, check if it's the best so far, otherwise discard it
                    # Save the n-best if it's too short or if it's more probably than the nth one
                    if len(n_best) < beam_width or prob > n_best[beam_width - 1][1]:
                        n_best.append([seq, prob])
                        # Sort by probability - highest first
                        n_best.sort(key=lambda x: -x[1])
                        del n_best[beam_width:]

        if not n_best:
            # This happens when we've reached max_tokens without any of the options terminating
            logger.warning(f'{prompt=}\tReached {max_tokens} without terminating')
            n_best = input_ids
        return n_best


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


def load_model(config):
    create_model(config)
    model = create_model(config)
    cp_file = config['PATH'] + 'best_mamba_model.cp'
    # Do not care about optimizer or training state
    checkpoint = torch.load(cp_file)
    # Load the model state dictionary
    model.load_state_dict(checkpoint['model_state'])
    return model


def create_model(config):
    model_params = {'model': config['model'], 'd_model': config['d_model'],
                    'n_layers': config['n_layers'], 'expand_factor': config['expand_factor']}
    model = G2pModel(model_params).to(device)
    return model


def main():
    logger.info(f'Device used: {device}')
    # Run a test on the generate function for a trained network
    path = "C:\\Users\\Richard\\Repository\\ddg2p\\experiments\\experiment_wiki_en\\"
    model_params = {'model': 'mamba', 'd_model': 256, 'n_layers': 2, 'expand_factor': 4, 'PATH': path}
    model = load_model(model_params).to(device)

    # Save the embedding weights to a CSV file
    with open('embedding_weights.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Token'] + [f'Dim_{i}' for i in range(256)])
        # Header row
        embedding_weights = model.embedding.weight.data.cpu().numpy()
        for idx, row in enumerate(embedding_weights):
            csvwriter.writerow([idx] + row.tolist())


    # TODO: This is mostly duplicate code - refactor this at some stage before the original code changes too much
    ipa_data = IpaDataset("C:\\Users\\Richard\\Repository\\g2p_data\\wikipron\\data\\scrape\\tsv\\", 'de_data.csv',
                          ['deu_latn_broad'], max_length=124, remove_spaces=True)

    # Split into train/test/valid subsets
    train_split = 0.8
    test_split = 0.1
    validation_split = 1. - train_split - test_split
    train_subset, test_subset, valid_subset = torch.utils.data.random_split(ipa_data,
                                                                            [train_split, test_split, validation_split],
                                                                            generator=torch.Generator().manual_seed(1))

    start_time = time.perf_counter()
    logger.info(f'Testing starts at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    language_cm, total_per = test_on_subset(test_subset, model)
    total_ler = 1. - language_cm.true_positive_rate()
    testing_elapsed_time = time.perf_counter() - start_time
    logger.info(f'Testing finished at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\t{testing_elapsed_time=:.4f}')
    logger.info(f'{language_cm}')
    for k, v in total_per:
        per = sum(v) / len(v)
        total_wer = [1 if p > 0 else 0 for p in v]
        wer = sum(total_wer) / len(total_wer)
        logger.info(f'{k}\t{total_ler=}\t{wer=}, {per=}')


if __name__ == '__main__':
    main()
