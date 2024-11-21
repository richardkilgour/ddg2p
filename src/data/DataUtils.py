import logging
import random

import pycountry
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchmetrics.text import WordErrorRate

from src.data.BucketBatchSampler import BucketBatchSampler
from src.data.ConfusionMatrix import ConfusionMatrix
from src.data.DataConstants import PAD, EOS, SEP

logger = logging.getLogger(__name__)


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
    # Step 1: Move the tensor to the CPU and convert it to byte type
    # Step 2: Convert the byte tensor to a NumPy array
    # Step 3: Convert the NumPy array to a byte string
    # Step 4: Convert the byte string to a UTF-8 string
    byte_string = tensor.cpu().byte().numpy().tobytes()
    try:
        s = byte_string.decode('utf-8').rstrip(PAD).rstrip(EOS)
    except UnicodeDecodeError:
        s = byte_string.decode('utf-8', errors='replace').rstrip(PAD).rstrip(EOS)
        logger.warning(f'invalid utf8: {s}')
    return s


def separate_phonemes(phonemes):
    """Space delimit all the phonemes so that they can be treated as 'words' and ready for PER calculation"""
    char_list = list(phonemes)
    # Merge combining diacritics et al.
    # U+02B0 through U+02FF are Spacing Modifier Letters
    # U+0300 through U+036F are combining diacritics, including
    # U+035C through U+0362 which tie two characters
    combined_list = []

    i = 0
    while i < len(char_list):
        if i < len(char_list) - 1 and '\u02B0' <= char_list[i + 1] <= '\u036F':
            # Handle pathological edge case where double combination is the last character
            if i < len(char_list) - 2 and '\u035C' <= char_list[i + 1] <= '\u0362':
                # This is followed by a double combining character
                combined_list.append(char_list[i] + char_list[i + 1] + char_list[i + 2])
                del char_list[i + 1]
                del char_list[i + 1]
            else:
                # This is followed by a single combining character
                combined_list.append(char_list[i] + char_list[i + 1])
                del char_list[i + 1]
        else:
            combined_list.append(char_list[i])
        i += 1
    return ' '.join(combined_list)


def calculate_per(reference, hypothesis):
    ref_words = separate_phonemes(reference)
    hyp_words = separate_phonemes(hypothesis)
    # Treating each phoneme as a 'word'
    per = WordErrorRate()(ref_words, hyp_words).item()
    return per


def get_metrics(words, target_languages, target_phonemes, model, beam_width=1):
    """
    For a given data subset, return:
    Confusion Matrix for Language guessing
    Dictionary of (List of) Phoneme Error Rates per language. eg: {'en_UK': [0., 0., 0.75, 0.54, 0.62]
    """
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
            out.append(tensor_to_utf8(beam[0][0]))

    # Initialise the confusion Matrix
    unique_languages = list(set(target_languages))
    confusion_matrix = ConfusionMatrix(unique_languages)

    language_per = dict.fromkeys(unique_languages, [])

    for t_lan, ortho, t_phon, output, g_out in zip(target_languages, words, target_phonemes, out, greedy):
        try:
            lan, ortho_, phon = parse_output(output)

            confusion_matrix.update(lan, t_lan)

            # Phoneme Error rate (as a proportion of the word length)
            per = calculate_per(phon, t_phon)
            # Word error if any phonemes are incorrect
            language_per[t_lan].append(per)

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
        except Exception as e:
            # Early networks fail this. One epoch seems enough to get (mostly) valid UTF-8
            logger.error(f"An exception occurred for {output}: {type(e).__name__} - {e}")
            confusion_matrix.update('ERROR', t_lan)
            language_per[t_lan].append(1.)
    return confusion_matrix, language_per


def parse_output(out):
    eos_pos = out.index(EOS)
    sep_pos = out.index(SEP)
    ortho = out[1:eos_pos]
    lan = out[eos_pos + 1:sep_pos]
    phon = out[sep_pos + 1:].strip(EOS).strip(PAD)
    return lan, ortho, phon


def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key].extend(value)
        else:
            merged_dict[key] = value
    return merged_dict


def test_on_subset(test_subset, model, beam_width=1, with_language=False):
    if with_language:
        bucket_sampler = BucketBatchSampler(test_subset, batch_size=64, delimiter=SEP)
    else:
        bucket_sampler = BucketBatchSampler(test_subset, batch_size=64, delimiter=EOS)
    dataloader = DataLoader(test_subset, batch_sampler=bucket_sampler, collate_fn=pad_collate, pin_memory=True)
    language_cm = None
    total_per = {}
    for source, _ in dataloader:
        languages, words, phonemes = zip(*[parse_output(tensor_to_utf8(item)) for item in source])
        lan_cm, per = get_metrics(words, languages, phonemes, model, beam_width)
        if language_cm:
            language_cm.merge(lan_cm)
        else:
            language_cm = lan_cm
        total_per = merge_dictionaries(total_per, per)
    return language_cm, total_per


def main():
    phon = 'beːtaːni̯ən'
    g_phon = 'beːtaːni̯ə'
    t_phon = 'beːtaːni̯ən'
    logger.info(f'{calculate_per(t_phon, phon)=}')
    logger.info(f'{calculate_per(t_phon, g_phon)=}')
    logger.info(f'{calculate_per(g_phon, t_phon)=}')


if __name__ == '__main__':
    main()
