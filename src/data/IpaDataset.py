import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.utils import BOS, EOS, PAD, SEP, string_to_class


def load_file_into_dataframe(data_path: str):
    data = []

    # Extract the file name (without the extension) == language code
    file_name = os.path.basename(data_path).split('.')[0]

    print(f'Processing {file_name}')

    # Read the file into a temporary DataFrame. 'nan' should be read in as 'nan', not as NaN
    df = pd.read_csv(data_path, delimiter='\t', header=None, keep_default_na=False, na_values=['_'])

    print(f'{df.size=}')

    # Iterate through each row in the DataFrame
    for i, row in df.iterrows():
        index = row[0]
        entries = row[1].split(', ')
        for enum, entry in enumerate(entries):
            data.append({
                'Language': file_name,
                'Ortho': index,
                'Pref': enum,
                'Phon': entry
            })

    print(f'Read {i} rows from {file_name}')
    return data


def format_input_output(ortho, lang, phono):
    input_data = BOS + ortho + EOS + lang + SEP + phono
    target_data = PAD * (1 + len(ortho.encode('utf-8'))) + lang + SEP + phono + EOS
    return input_data, target_data


def total_length(row):
    return len((row['Ortho'] + row['Language'] + row['Phon']).encode('utf-8'))


# TODO: Train/test/validate splits
class IpaDataset(Dataset):
    def __init__(self, datapath, csv_filename, languages=None, dummy_data=False, max_length=None):
        # TODO?: Ingest the data - https://github.com/open-dict-data/ipa-dict
        self.path = datapath
        self.filename = csv_filename

        if dummy_data:
            ortho = ['she', 'had', 'your', 'dark', 'suit', 'in', 'greasy', 'wash', 'water', 'all', 'year',
                     'she had your', 'your dark suit', 'suit in greasy', 'greasy wash water', 'water all year']
            phono = [u"ˈʃi", u"ˈhæd", u"ˈjɔɹ", u"ˈdɑɹk", u"ˈsut", u"ˈɪn", u"ˈɡɹisi", u"ˈwɑʃ", u"ˈwɔtɝ", u"ˈɔɫ", u"ˈjɪɹ",
                     u"ˈʃi ˈhæd ˈjɔɹ", u"ˈjɔɹ ˈdɑɹk ˈsut", u"ˈsut ˈɪn ˈɡɹisi", u"ˈɡɹisi ˈwɑʃ ˈwɔtɝ", u"ˈwɔtɝ ˈɔɫ ˈjɪɹ"]
            lang = ['en_US'] * len(ortho)
            pref = [1] * len(ortho)
            self.data = {'Ortho': ortho, 'Phono': phono, 'Language': lang, 'Pref': pref}
            return

        full_path = self.path + self.filename
        # Recreate the data scv file if it does not already exist
        if not os.path.isfile(full_path):
            print(f'Recreating {self.filename}. This may take a while...')
            final_df = pd.DataFrame(columns=['Language', 'Ortho', 'Pref', 'Phon'])

            # Each .txt file is the phonemes for a given language
            for dir_path, dir_names, filenames in os.walk(self.path):
                for file_name in filenames:
                    lan, ext = file_name.split('.')
                    if ext == 'txt' and (languages is None or lan in languages):
                        new_df = pd.DataFrame(load_file_into_dataframe(self.path + file_name))
                        # Remove any item longer than the max_length
                        if max_length:
                            new_df = new_df[new_df.apply(total_length, axis=1) <= max_length]
                        final_df = pd.concat([final_df, new_df]).reset_index(drop=True)
            print(final_df)

            # Export the DataFrame to a CSV file
            final_df.to_csv(full_path, index=False)
        self.data = pd.read_csv(full_path)
        # TODO: (properly) NaN should actually be the string 'nan'
        self.data.Ortho = self.data.Ortho.fillna('nan')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a byte string. This still needs to be embedded and padded
        item = self.data.iloc[idx]
        source, targets = format_input_output(item['Ortho'], item['Language'], item['Phon'])
        x_enc = string_to_class(source)
        t_enc = string_to_class(targets)
        return torch.tensor(x_enc, dtype=torch.long), torch.tensor(t_enc, dtype=torch.long)
