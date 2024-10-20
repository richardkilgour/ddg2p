import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig


class ddg2pModel(nn.Module):
    def __init__(self, params):
        """Full Mamba model."""
        super().__init__()
        self.embedding = nn.Embedding(256, 256)
        # Initialize the embedding weights to be an identity matrix
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(256, 256))

        d_model = params['d_model']
        n_layers = params['n_layers']
        if params['model'] == 'mamba':
            config = MambaConfig(d_model=d_model, n_layers=n_layers, expand_factor=4)
            self.recurrence = Mamba(config)
        elif params['model'] == 'gru':
            self.recurrence = nn.GRU(256, 1024, n_layers)
        self.probs = nn.Linear(d_model, 256, bias=False)

    def forward(self, input_ids):
        # Do the embedding
        x = self.embedding(input_ids)
        x = self.recurrence(x)
        return self.probs(x)
