import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig
from torch.autograd import profiler

from src.data.utils import string_to_class, BOS, EOS, PAD

PROFILING = False


class ddg2pModel(nn.Module):
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

    def forward(self, input_ids):
        # TODO: This is nasty
        # Do the embedding
        if PROFILING:
            with profiler.record_function('EMBEDDING'):
                x = self.embedding(input_ids)
        else:
            x = self.embedding(input_ids)
        if PROFILING:
            with profiler.record_function(self.model.upper()):
                x = self.recurrence(x)
        else:
            x = self.recurrence(x)
        if PROFILING:
            with profiler.record_function('LINEAR_OUT'):
                x = self.probs(x)
        else:
            x = self.probs(x)
        return x

    def generate(self,
                 prompt: str,
                 device: torch.device,
                 n_tokens_to_gen: int = 50,
                 ):
        self.eval()

        b_prompt = BOS + prompt + EOS
        b_encoded = string_to_class(b_prompt)

        # Convert byte data to a numpy array
        # TODO: Device should not be here (or even a param)
        input_ids = torch.unsqueeze(torch.tensor(b_encoded, dtype=torch.long), 0).to(device)

        for token_n in range(n_tokens_to_gen):
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
