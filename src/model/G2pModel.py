import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Subset
from mambapy.mamba import Mamba, MambaConfig

from src.data.IpaDataset import IpaDataset
from src.data.utils import string_to_class, test_on_subset
from src.data.DataConstants import PAD, BOS, EOS, PROFILING, logger, device

if PROFILING:
    from torch.autograd import profiler


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


def tensor_to_string(net_out):
    # Step 1: Create a tensor and convert it to byte type
    byte_tensor = net_out.byte()
    # Step 2: Convert the byte tensor to a NumPy array
    byte_array = byte_tensor.cpu().numpy()
    # Step 3: Convert the NumPy array to a byte string
    byte_string = byte_array.tobytes()
    # Step 4: Convert the byte string to a UTF-8 string
    try:
        output_completions = byte_string.decode('utf-8')
    except UnicodeDecodeError:
        logger.warning(f'invalid utf8: {byte_string=}')
        output_completions = byte_string.decode('utf-8', errors='replace')
    return output_completions.rstrip(PAD)


class G2pModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(256, 256)
        # Initialize the embedding weights to be an identity matrix
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(256, 256))

        d_model = params['d_model']
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

        return [tensor_to_string(ids.byte()) for ids in encoded_prompts]

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
        self.eval()

        b_prompt = BOS + prompt + EOS
        b_encoded = string_to_class(b_prompt)

        # Convert byte data to a numpy array, batch size 1
        input_ids = [[torch.tensor(b_encoded, dtype=torch.long).to(device), 1.]]

        best_terminated_sequence = None
        best_terminated_probability = 0.
        # List of candidate sequences, and their probabilities
        for _ in range(max_tokens):
            input_ids = self._get_next_beam(input_ids, beam_width)
            # Scan the list backwards so we can safely remove items
            for i in range(len(input_ids) - 1, -1, -1):
                seq, prob = input_ids[i]
                # We can stop tracking this if we've already found a better candidate, as probability will not increase!
                # Or if the sequence has hit a termination character
                if prob < best_terminated_probability or seq[-1] == ord(EOS) or seq[-1] == ord(PAD):
                    # Remove it from the beam search
                    input_ids.pop(i)
                    # If it's the best so far, remember it
                    if prob > best_terminated_probability:
                        best_terminated_sequence = seq
                        best_terminated_probability = prob

        # Should have a list of sequences that have terminated - convert to string
        return tensor_to_string(best_terminated_sequence)


def main():
    logger.info(f'Device used: {device}')
    # Run a test on the generate function for a trained network
    path = "C:\\Users\\Richard\\Repository\\ddg2p\\experiments\\experiment_wiki_en\\mamba_model_de.ckp"
    model_params = {'model': 'mamba', 'd_model': 256, 'n_layers': 2, 'expand_factor': 4}

    model = G2pModel(model_params).to(device)
    model.load_state_dict(torch.load(path))
    for out in model.generate(['Abstoßung', 'hakelig', 'jahrhundertelang', 'Umzug']):
        logger.info(out)

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
    total_ler, total_per, total_wer = test_on_subset(test_subset, model)

    testing_elapsed_time = time.perf_counter() - start_time
    logger.info(f'Testing finished at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\t{testing_elapsed_time=:.4f}')
    logger.info(f'{total_ler=}\t{total_wer=}, {total_per=}')


if __name__ == '__main__':
    main()
