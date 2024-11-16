import random
import pycountry
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
from torchmetrics.text import WordErrorRate

from src.data.BucketBatchSampler import BucketBatchSampler
from src.data.DataConstants import PAD, EOS, SEP, logger


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    assert (x_lens == y_lens)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=ord(PAD))
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=ord(PAD))
    return xx_pad, yy_pad


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
        out = []
        # For now, send them one at a time and take the single best result
        for w in words:
            beam = model.generate_beam(w, beam_width)
            # The first beam is the most likely
            out.append(tensor_to_string(beam[0][0]))

    language_errors = 0
    cumulative_per = 0.
    word_errors = 0
    target_languages = [subset.dataset.data.iloc[i]['Language'] for i in subset.indices]
    target_phonemes = [subset.dataset.data.iloc[i]['Phon'] for i in subset.indices]

    for t_lan, ortho, t_phon, output, g_out in zip(target_languages, words, target_phonemes, out, greedy):
        try:
            lan, ortho_, phon = parse_output(output)
            language_errors += t_lan != lan

            # Phoneme Error rate (as a proportion of the word length)
            per = calculate_per(phon, t_phon)
            # Word error if any phonemes are incorrect
            cumulative_per += per
            word_errors += per > 0.

            # Let's compare them!!!
            if g_out != output:
                g_lan, g_ortho, g_phon = parse_output(g_out)
                g_per = calculate_per(phon, g_phon)
                logger.info(
                    f'{ortho=} BEAM got {"BETTER" if per < g_per else "WORSE" if per > g_per else "DIFFERENT"} '
                    f'result\t{phon=}\t{g_phon=}\t{t_phon=}')

            # Print every 100 words randomly
            if random.randint(1, 100) == 60:
                logger.info(f'{ortho=}\t{lan=}\t{phon=}\t{t_lan=}\t{t_phon=}\t{per=}')
        except:
            # Early networks fail this. One epoch seems enough to get (mostly) valid UTF-8
            logger.error(f'PARSE FAILED: {output}')
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
