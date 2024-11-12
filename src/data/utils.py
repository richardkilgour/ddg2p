import random
import pycountry
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from torchmetrics.text import WordErrorRate

PAD = u'\x00'  # ASCII Null
BOS = u'\x02'  # ASCII Start of Text
EOS = u'\x03'  # ASCII End of Text
FIN = u'\x04'  # ASCII End of Transmission
SEP = u'\x1d'  # ASCII Group Seperator (Separates language code from phonemes)


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    assert (x_lens == y_lens)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=ord(PAD))
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=ord(PAD))
    return xx_pad, yy_pad


class BucketBatchSampler(Sampler):
    def __init__(self, data, batch_size):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.buckets = self.create_buckets()

    def create_buckets(self):
        # Create buckets based on length
        length_buckets = {}
        for i, item in enumerate(self.data):
            length = len(item[0])
            if length not in length_buckets:
                length_buckets[length] = []
            length_buckets[length].append(i)

        # Sort buckets based on size from smallest to largest
        sorted_buckets = dict(sorted(length_buckets.items()))

        # Create batches
        sorted_items = []
        for v in sorted_buckets.values():
            sorted_items.extend(v)

        batches = []
        for i in range(0, len(sorted_items), self.batch_size):
            batches.append(sorted_items[i:i + self.batch_size])

        # Shuffle all batches
        random.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self.buckets:
            yield batch

    def __len__(self):
        return len(self.buckets)


def iso3_to_iso2(iso3_code: str) -> str:
    language = pycountry.languages.get(alpha_3=iso3_code)
    return language.alpha_2 if language else iso3_code


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


def calculate_per(reference, hypothesis):
    ref_words = list(reference)
    hyp_words = list(hypothesis)
    return WordErrorRate()(ref_words, hyp_words).item()


def get_metrics(word, model, device, beam_width=1):
    """
    For a given word, return the:
    Language is 100% correct -> Bool
    Phoneme Error Rate -> float
    """
    greedy = model.generate(word['Ortho'], device)
    if beam_width == 1:
        out = greedy
    else:
        out = model.generate_beam(word['Ortho'], device, beam_width)

    correct_language = False
    per = 1.
    targ_lan = word['Language']
    targ_phn = word['Phon']

    try:
        lan, ortho, phon = parse_output(out)

        correct_language = targ_lan == lan

        # Phoneme Error rate (as a proportion of the word length)
        per = calculate_per(phon, targ_phn)

        # Let's compare them!!!
        if greedy != out:
            g_lan, g_ortho, g_phon = parse_output(greedy)
            g_per = calculate_per(phon, g_phon)
            print(
                f'{word["Ortho"]=}, BEAM got {"BETTER" if per < g_per else "WORSE" if per > g_per else "DIFFERENT"} '
                f'result\t{phon=}\t{g_phon=}\t{targ_phn=}')

        # Print every 100 words randomly
        if random.randint(1, 100) == 60:
            print(f'{ortho=}\t{lan=}\t{phon=}\t{targ_lan=}\t{targ_phn=}\t{per=}')
    except:
        # Early networks fail this. One epoch seems enough to get valid UTF-8
        print(f'PARSE FAILED: {out}')
    return correct_language, per


def parse_output(out):
    eos_pos = out.index(EOS)
    sep_pos = out.index(SEP)
    # TODO: Splitting on bytes is a pain
    ortho = out[1:eos_pos]
    lan = out[eos_pos + 1:sep_pos]
    phon = out[sep_pos + 1:-1]
    return lan, ortho, phon


def test_on_subset(subset, model, device, beam_width=3):
    """For a given subset, return average language, word and phoneme error rates. Lower is better."""
    total_ler = 0
    total_wer = 0
    total_per = 0
    subset_len = len(subset)
    for i in subset.indices:
        correct_language, per = get_metrics(subset.dataset.data.iloc[i], model, device, beam_width)
        total_ler += 0 if correct_language else 1
        total_wer += 0 if per == 0. else 1
        total_per += per
    return total_ler / subset_len, total_wer / subset_len, total_per / subset_len
