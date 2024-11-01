import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.text import WordErrorRate

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    assert (x_lens == y_lens)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=ord(PAD))
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=ord(PAD))
    return xx_pad, yy_pad


def string_to_class(s):
    return [int(b) for b in bytearray(s.encode('utf-8'))]


def pad_batch(batch, pad):
    max_length = max(len(x) for x in batch)
    return [b + [pad] * (max_length - len(b)) for b in batch]


def encode_and_pad_batch(batch, pad):
    encoded_batch = [string_to_class(b) for b in batch]
    return pad_batch(encoded_batch, pad)


def tensor_to_utf8(tensor):
    byte_string = tensor.numpy().tobytes()
    try:
        s = byte_string.decode('utf-8')
    except UnicodeDecodeError:
        print(f'invalid utf8: {byte_string=}')
        s = byte_string
    return s


def byte_array_to_one_hot(z):
    # OBSOLETE?
    # TODO: use torch.nn.functional.one_hot
    def one_hot_encode(byte_array, num_classes=256):
        one_hot = torch.zeros(len(byte_array), num_classes)
        for i, byte in enumerate(byte_array):
            one_hot[i][byte] = 1
        return one_hot

    # One-hot encode each byte array
    one_hot = [one_hot_encode(byte_array) for byte_array in z]
    # Stack tensors to create a batch
    return torch.stack(one_hot)


def encode_and_pad(x_raw, y_raw, pad=0):
    # OBSOLETE?
    max_len = 0
    x = []
    y = []
    for x_, y_ in zip(x_raw, y_raw):
        x_ = x_.encode('utf-8')
        y_ = y_.encode('utf-8')
        # x.append(BOS + x_ + EOS + b'en_US' + SEP + y_)
        x.append(x_)
        # Y is offset by one, and has no inputs
        # Does it need to be masked? Yes.
        # y.append(PAD * (1 + len(x_)) + b'en_US' + SEP + y_ + EOS)
        y.append(bytearray([e + 1 for e in x_]))
        max_len = max(max_len, len(x[-1]))

    # Padding
    x_padded = [z.ljust(max_len, pad) for z in x]
    y_padded = [z.ljust(max_len, pad) for z in y]

    return x_padded, y_padded


def calculate_per(reference, hypothesis):
    ref_words = list(reference)
    hyp_words = list(hypothesis)
    return WordErrorRate()(ref_words, hyp_words).item()


PAD = u'\x00'  # ASCII Null
BOS = u'\x02'  # ASCII Start of Text
EOS = u'\x03'  # ASCII End of Text
FIN = u'\x04'  # ASCII End of Transmission
SEP = u'\x1d'  # ASCII Group Seperator (Separates language code from phonemes)


def get_metrics(word, model, device):
    """
    For a given word, see if the:
    Language is 100% correct -> Bool
    Phonemes are 100% correct -> Bool
    Phoneme Error Rate -> float
    """
    out = model.generate(word['Ortho'], device)
    try:
        EOS_pos = out.index(EOS)
        SEP_pos = out.index(SEP)
        # TODO: Splitting on bytes is a pain
        ortho = out[1:EOS_pos]
        lan = out[EOS_pos + 1:SEP_pos]
        phon = out[SEP_pos + 1:-1]

        # TODO: Was the language correct?
        targ_lan = word['Language']
        correct_language = targ_lan == lan
        # TODO: Was the phoneme sequence correct? WER
        targ_phn = word['Phon']
        correct_phoneme = targ_phn == phon
        # TODO: Were the phonemes correct? PER - this value is so wrong. Do not trust it.
        PER = calculate_per(phon, targ_phn)
        if random.randint(1, 100) == 60:
            print(f'{ortho=}\t{lan=}\t{phon=}\t{targ_lan=}\t{targ_phn=}\t{PER=}')
        return correct_language, correct_phoneme, PER
    except:
        # Early networks fail this. One epoch seems enough to get valid UTF-8
        if random.randint(1, 100) == 60:
            print(f'PARSE FAILED: {out}')
    return False, False, 0.


def test_on_subset(subset, model, device):
    correct_language = 0
    correct_phoneme = 0
    total_per = 0
    counter = 0
    subset_len = len(subset)
    for i in subset.indices:
        counter += 1
        # TODO: I made this real ugly for some reason
        correct_language_, correct_phoneme_, total_per_ = get_metrics(subset.dataset.data.iloc[i], model, device)
        correct_language += correct_language_
        correct_phoneme += correct_phoneme_
        total_per += total_per_
    return correct_language/subset_len, correct_phoneme/subset_len, total_per/subset_len
