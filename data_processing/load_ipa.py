import os
import pandas as pd


def load_file_into_dataframe(file_path):
    data = []

    # Extract the file name (without the extension)
    file_name = os.path.basename(file_path).split('.')[0]

    print(f'Processing {file_name}')

    # Read the file into a temporary DataFrame
    df = pd.read_csv(file_path, delimiter='\t', header=None)

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


class IpaLoader:
    def __init__(self, datapath, filename):
        # Ingest the data - https://github.com/open-dict-data/ipa-dict
        self.path = datapath
        self.filename = filename

    def load_or_create(self):
        full_path = self.path + self.filename
        if not os.path.isfile(full_path):
            print(f'Recreating {self.filename}. This may take a while...')
            final_df = pd.DataFrame(columns=['Language', 'Ortho', 'Pref', 'Phon'])

            # Each .txt file is the phonemes for a given language
            for dir_path, dir_names, filenames in os.walk(self.path):
                for file_name in filenames:
                    if file_name[-4:] == '.txt':
                        new_df = pd.DataFrame(load_file_into_dataframe(full_path))
                        final_df = pd.concat([final_df, new_df]).reset_index(drop=True)
            print(final_df)

            # Export the DataFrame to a CSV file
            final_df.to_csv(full_path, index=False)
        return pd.read_csv(full_path)
