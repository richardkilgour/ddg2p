import logging
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.DataUtils import string_to_class, iso3_to_iso2
from src.data.DataConstants import PAD, BOS, EOS, SEP

logger = logging.getLogger(__name__)

def wikidata_to_ISO2(name):
    """Wikidata style file name is language_script[_country]_transcription_filtered"""
    components = name.split('_')
    language = iso3_to_iso2(components[0])
    if components[2] in ['broad', 'narrow']:
        return language
    region = components[2].upper()
    return language + '_' + region


def load_file_into_dataframe(data_path: str, remove_spaces=False):
    data = []

    # Extract the file name (without the extension) == language code
    file_name, ext = os.path.basename(data_path).split('.')

    logger.info(f'Processing {file_name}')

    # Read the file into a temporary DataFrame. 'nan' should be read in as 'nan', not as NaN
    df = pd.read_csv(data_path, delimiter='\t', header=None, keep_default_na=False, na_values=['_'])

    logger.debug(f'{df.size=}')

    if ext == 'tsv':
        # TODO: Identify wikidata formated names better. Maybe a parameter?
        language_code = wikidata_to_ISO2(file_name)
    else:
        language_code = file_name

    # Iterate through each row in the DataFrame
    for i, row in df.iterrows():
        index = row[0]
        entries = row[1].split(', ')
        for enum, entry in enumerate(entries):
            data.append({
                'Language': language_code,
                'Ortho': index,
                'Pref': enum,
                # Remove the phoneme markers
                'Phon': entry.replace(" ", "").strip('/') if remove_spaces else entry.strip('/')
            })

    logger.info(f'Read {i} rows from {file_name}')
    return data


def format_input_output(ortho, lang, phono):
    input_data = BOS + ortho + EOS + lang + SEP + phono
    target_data = PAD * (1 + len(ortho.encode('utf-8'))) + lang + SEP + phono + EOS
    return input_data, target_data


def total_length(row):
    return len((row['Ortho'] + row['Language'] + row['Phon']).encode('utf-8'))


class IpaDataset(Dataset):
    def __init__(self, datapath, out_filename, splits, languages=None, max_length=None, remove_spaces=False):
        self.path = datapath
        self.filename = out_filename

        full_path = self.path + self.filename
        # Recreate the data scv file if it does not already exist
        if not os.path.isfile(full_path):
            logger.info(f'Recreating {self.filename}. This may take a while...')
            self.data = pd.DataFrame(columns=['Language', 'Ortho', 'Pref', 'Phon'])

            # Each .txt file is the phonemes for a given language
            for dir_path, dir_names, filenames in os.walk(self.path):
                for file_name in filenames:
                    lan, ext = file_name.split('.')
                    if ext in ['tsv', 'txt'] and (languages is None or lan in languages):
                        new_df = pd.DataFrame(load_file_into_dataframe(self.path + file_name, remove_spaces))
                        # Remove any item longer than the max_length
                        if max_length:
                            new_df = new_df[new_df.apply(total_length, axis=1) <= max_length]
                        self.data = pd.concat([self.data, new_df]).reset_index(drop=True)
            # TODO: (properly) NaN should actually be the string 'nan'
            self.data.Ortho = self.data.Ortho.fillna('nan')
            logger.debug(self.data)
            self.train_subset, self.test_subset, self.valid_subset = torch.utils.data.random_split(self, splits,
                                                                                                   generator=torch.Generator().manual_seed(
                                                                                                       1))
            # Export the DataFrame to a CSV file
            torch.save(
                {'data': self.data, 'training': self.train_subset, 'testing': self.test_subset, 'validation': self.valid_subset},
                full_path)
        all_data = torch.load(full_path)
        self.data = all_data['data']
        self.train_subset = all_data['training']
        self.test_subset = all_data['testing']
        self.valid_subset = all_data['validation']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a byte string. This still needs to be embedded and padded
        item = self.data.iloc[idx]
        source, targets = format_input_output(item['Ortho'], item['Language'], item['Phon'])
        x_enc = string_to_class(source)
        t_enc = string_to_class(targets)
        return torch.tensor(x_enc, dtype=torch.long), torch.tensor(t_enc, dtype=torch.long)
