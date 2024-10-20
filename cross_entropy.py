"""
Simple example of cross entropy with all the tensors the right size and stuff
input: UTF8 String
output: UTF8 String
internally, it turns the string into the byte representation and feeds it through the network
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch import argmax

from data_processing.utils import encode_and_pad_batch, tensor_to_utf8


class SimpleNLPModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNLPModel, self).__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.dense = nn.Linear(num_classes, num_classes)
        self.output_layer = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dense(x)
        x = self.output_layer(x)
        return nn.functional.softmax(x, dim=2)


# Parameters
num_classes = 256

# Initialize the model, loss function, and optimizer
model = SimpleNLPModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example data - some words and target is rot1 a ->b, b->c etc
x_raw = ['she', u"ˈʃi", 'had', 'your', 'dark', 'suit', 'in', 'greasy', 'wash', 'water', 'all', 'year', ]
y_raw = [''.join([chr(ord(c) + 1) for c in w]) for w in x_raw]

padded_in = encode_and_pad_batch(x_raw, 0)
input_data = torch.Tensor(padded_in).int()

padded_out = encode_and_pad_batch(y_raw, 0)
target_data = torch.Tensor(padded_out).long()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_data)

    # Compute loss
    loss = criterion(outputs.permute(0, 2, 1), target_data)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed!")


for i, t in zip(input_data, target_data):
    # TODO: no grad
    out = model(i.unsqueeze(0))
    classes = argmax(out, dim=2).squeeze(0).to(dtype=torch.uint8)
    print(f'Pred: {tensor_to_utf8(classes)}')
    print(f'Targ: {tensor_to_utf8(t.to(dtype=torch.uint8))}')
