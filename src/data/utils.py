import logging
import random
import pycountry
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, Subset
from torchmetrics.text import WordErrorRate

PAD = u'\x00'  # ASCII Null
BOS = u'\x02'  # ASCII Start of Text
EOS = u'\x03'  # ASCII End of Text
FIN = u'\x04'  # ASCII End of Transmission
SEP = u'\x1d'  # ASCII Group Seperator (Separates language code from phonemes)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    assert (x_lens == y_lens)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=ord(PAD))
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=ord(PAD))
    return xx_pad, yy_pad


class BucketBatchSampler(Sampler):
    def __init__(self, data, batch_size, test_set):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.buckets = self.create_buckets(test_set)

    def create_buckets(self, test_set=False):
        # Create buckets based on length
        length_buckets = {}
        for i in self.data.indices:
            item = self.data.dataset[i][0]
            # For testing, we only care about the length of the orthography (up to the EOS character)
            if test_set:
                length = torch.nonzero(item == ord(EOS)).item()
            else:
                length = len(item)
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
        logger.warning(f'invalid utf8: {byte_string=}')
        s = byte_string
    return s


def calculate_per(reference, hypothesis):
    ref_words = list(reference)
    hyp_words = list(hypothesis)
    # Treating each phoneme as a 'word'
    per = WordErrorRate()(ref_words, hyp_words).item()
    assert 0. <= per <= 1.
    return per


def get_metrics(subset, model, beam_width=1):
    """
    For a given data subset, return:
    Language is correct %
    Word Error Rate %
    Average Phoneme Error Rate
    """
    words = [subset.dataset.data.iloc[i]['Ortho'] for i in subset.indices]
    greedy = model.generate(words)
    if beam_width == 1:
        out = greedy
    else:
        # TODO: Need to make this batch, too!
        out = model.generate_beam(words, beam_width)

    language_errors = 0
    cumulative_per = 0.
    word_errors = 0
    target_languages = [subset.dataset.data.iloc[i]['Language'] for i in subset.indices]
    target_phonemes = [subset.dataset.data.iloc[i]['Phon'] for i in subset.indices]

    for t_lan, ortho, t_phon, output in zip(target_languages, words, target_phonemes, out):
        try:
            lan, ortho_, phon = parse_output(output)
            language_errors += t_lan != lan

            # Phoneme Error rate (as a proportion of the word length)
            per = calculate_per(phon, t_phon)
            # Word error if any phonemes are incorrect
            cumulative_per += per
            word_errors += per > 0.

            # TODO: Let's compare them!!!
            if greedy != out and False:
                g_lan, g_ortho, g_phon = parse_output(greedy)
                g_per = calculate_per(phon, g_phon)
                logger.info(
                    f'{ortho=} BEAM got {"BETTER" if per < g_per else "WORSE" if per > g_per else "DIFFERENT"} '
                    f'result\t{phon=}\t{g_phon=}\t{t_phon=}')

            # Print every 100 words randomly
            if random.randint(1, 100) == 60:
                logger.info(f'{ortho=}\t{lan=}\t{phon=}\t{t_lan=}\t{t_phon=}\t{per=}')
        except:
            # Early networks fail this. One epoch seems enough to get (mostly) valid UTF-8
            logger.error(f'PARSE FAILED: {out}')
            language_errors += 1
            cumulative_per += 1.
            word_errors += 1
    return language_errors / len(subset), word_errors / len(subset), cumulative_per / len(subset)


def parse_output(out):
    eos_pos = out.index(EOS)
    sep_pos = out.index(SEP)
    ortho = out[1:eos_pos]
    lan = out[eos_pos + 1:sep_pos]
    phon = out[sep_pos + 1:-1]
    return lan, ortho, phon


def test_on_subset(test_subset, model, beam_width=1):
    bucket_sampler = BucketBatchSampler(test_subset, batch_size=64, test_set=True)
    total_ler = 0
    total_wer = 0
    total_per = 0
    for batch_indices in bucket_sampler.buckets:
        batch_subset = Subset(test_subset.dataset, batch_indices)
        ler, wer, per = get_metrics(batch_subset, model, beam_width)
        total_ler += ler
        total_wer += wer
        total_per += per
    total_ler /= len(bucket_sampler.buckets)
    total_wer /= len(bucket_sampler.buckets)
    total_per /= len(bucket_sampler.buckets)
    return total_ler, total_per, total_wer
