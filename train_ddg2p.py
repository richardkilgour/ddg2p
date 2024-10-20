import pandas as pd
import torch
from data_processing.load_ipa import IpaLoader

from data_processing.utils import string_to_class, pad_batch
from ddg2pmodel import ddg2pModel

# Define the network
d_in = 256  # Byte encoding
d_out = 256  # Byte decoding

PAD = u'\x00'  # ASCII Null
BOS = u'\x02'  # ASCII Start of Text
EOS = u'\x03'  # ASCII End of Text
FIN = u'\x04'  # ASCII End of Transmission
SEP = u'\x1d'  # ASCII Group Seperator (Separates language code from phonemes)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using CUDA')
else:
    device = torch.device("CPU")
    print('Using CPU')

phase = 'all'

if phase in ['load_data', 'all']:
    # Load the data
    ipa_data = IpaLoader('C:\\Users\\Richard\\Repository\\g2p_data\\ipa-dict\\data\\', 'all_data.csv').load_or_create()
    # ['Language', 'Ortho', 'Pref', 'Phon']
    # Ignore preference for now
    ortho = list(ipa_data['Ortho'])
    phono = list(ipa_data['Phon'])
    lang = list(ipa_data['Language'])
else:
    ortho = ['she', 'had', 'your', 'dark', 'suit', 'in', 'greasy', 'wash', 'water', 'all', 'year',
             'she had your', 'your dark suit', 'suit in greasy', 'greasy wash water', 'water all year']
    phono = [u"ˈʃi", u"ˈhæd", u"ˈjɔɹ", u"ˈdɑɹk", u"ˈsut", u"ˈɪn", u"ˈɡɹisi", u"ˈwɑʃ", u"ˈwɔtɝ", u"ˈɔɫ", u"ˈjɪɹ",
             u"ˈʃi ˈhæd ˈjɔɹ", u"ˈjɔɹ ˈdɑɹk ˈsut", u"ˈsut ˈɪn ˈɡɹisi", u"ˈɡɹisi ˈwɑʃ ˈwɔtɝ", u"ˈwɔtɝ ˈɔɫ ˈjɪɹ"]
    lang = ['en_US'] * len(ortho)

x_s = []
y_s = []

for i, t, l in zip(ortho, phono, lang):
    if l == 'en_US':
        # TODO: 'nan' is coming in as literal nan, not string 'nan'
        if pd.isnull(i):
            continue
        x_s.append(BOS + i + EOS + l + SEP + t)
        y_s.append(PAD * (1 + len(i)) + l + SEP + t + EOS)

x = [string_to_class(b) for b in x_s]
y = [string_to_class(b) for b in y_s]

x = pad_batch(x, 0)
y = pad_batch(y, 0)

batch_size = 128

input_tensors = torch.tensor([list(b) for b in x], dtype=torch.long)
output_tensors = torch.tensor([list(b) for b in y], dtype=torch.long)
# Reshape to (batch_size, 256)
input_tensors = input_tensors[:batch_size].to(device)
output_tensors = output_tensors[:batch_size].to(device)

# Print shapes to verify
print(f"{input_tensors.shape=}")
print(f"{output_tensors.shape=}")

params = {'model': 'mamba', 'd_model': 256, 'n_layers': 2}
net = ddg2pModel(params).to(device)

PATH = "C:\\Users\\Richard\\Repository\\ddg2p\\mamba_model.ckp"

if phase in ['train', 'all']:
    # Train it
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()

    running_loss = 0.
    last_loss = 0.

    for epoch in range(1000):
        # Make predictions for this batch
        outputs = net(input_tensors)

        # Compute the loss and its gradients
        loss = loss_fn(outputs.permute(0, 2, 1), output_tensors)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        loss.backward()
        # Adjust learning weights
        optimizer.step()

        print(f'{epoch=}; {loss=}')

    torch.save(net.state_dict(), PATH)
else:
    net.load_state_dict(torch.load(PATH, weights_only=False))


# print(str(net))


# Test it
def generate(model,
             prompt: str,
             n_tokens_to_gen: int = 50,
             ):
    model.eval()

    b_prompt = BOS + prompt + EOS
    b_encoded = string_to_class(b_prompt)

    # Convert byte data to a numpy array
    input_ids = torch.unsqueeze(torch.tensor(b_encoded, dtype=torch.long), 0).to(device)

    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]

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


# Define the tokenizer with a byte-level model

for w in ortho:
    print(generate(model=net, prompt=w))
print(generate(model=net, prompt="Barista"))
