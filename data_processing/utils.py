import torch


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
