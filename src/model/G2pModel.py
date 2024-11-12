import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig

from src.data.utils import string_to_class, BOS, EOS, PAD

PROFILING = False

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
            self.recurrence = nn.GRU(256, 1024, n_layers)
        self.probs = nn.Linear(d_model, 256, bias=False)

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
                 prompt: str,
                 device: torch.device,
                 max_tokens: int = 100,
                 ):
        self.eval()

        b_prompt = BOS + prompt + EOS
        b_encoded = string_to_class(b_prompt)

        # Convert byte data to a numpy array
        # TODO: Device should not be here (or even a param)
        input_ids = torch.unsqueeze(torch.tensor(b_encoded, dtype=torch.long), 0).to(device)

        for token_n in range(max_tokens):
            with torch.no_grad():
                indices_to_input = input_ids
                next_token_logits = self(indices_to_input)[:, -1]

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            next_indices = torch.argmax(probs, dim=-1)[:, None]
            next_value = int(next_indices.squeeze())

            input_ids = torch.cat([input_ids, next_indices], dim=1)
            if next_value == ord(EOS) or next_value == ord(PAD):
                break

        # Step 1: Create a tensor and convert it to byte type
        byte_tensor = input_ids[0].byte()
        # Step 2: Convert the byte tensor to a NumPy array
        byte_array = byte_tensor.cpu().numpy()
        # Step 3: Convert the NumPy array to a byte string
        byte_string = byte_array.tobytes()
        # Step 4: Convert the byte string to a UTF-8 string
        try:
            output_completions = byte_string.decode('utf-8')
        except UnicodeDecodeError:
            print(f'invalid utf8: {byte_string=}')
            output_completions = byte_string

        return output_completions

    def _get_next_beam(self, sequences, beam_width):
        # List of candidate sequences, and their probabilities
        new_seq = []
        # For every sequence, run the network, and grab the k most probable continuations
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
                      device: torch.device,
                      beam_width=5,
                      max_tokens: int = 100,
                      ):
        self.eval()

        b_prompt = BOS + prompt + EOS
        b_encoded = string_to_class(b_prompt)

        # Convert byte data to a numpy array, batch size 1
        # TODO: Device should not be here (or even a param)
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

        # Should have a list of sequences that have terminated

        # Step 1: Create a tensor and convert it to byte type
        byte_tensor = best_terminated_sequence.byte()
        # Step 2: Convert the byte tensor to a NumPy array
        byte_array = byte_tensor.cpu().numpy()
        # Step 3: Convert the NumPy array to a byte string
        byte_string = byte_array.tobytes()
        # Step 4: Convert the byte string to a UTF-8 string
        try:
            output_completions = byte_string.decode('utf-8')
        except UnicodeDecodeError:
            print(f'invalid utf8: {byte_string=}')
            output_completions = byte_string

        return output_completions
