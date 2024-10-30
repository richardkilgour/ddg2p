# Define the network
data_config = {
    'd_in': 256,  # Byte encoding
    'd_out': 256,  # Byte decoding
    'data_path': 'C:\\Users\\Richard\\Repository\\g2p_data\\ipa-dict\\data\\',
    'csv_name': 'def_data.csv',
    'batch_size': 64,
    'languages': ['en_UK', 'fr_FR', 'de']
}

model_config = {
    'model': 'mamba',
    'd_model': 256,
    'n_layers': 2,
    'PATH': "C:\\Users\\Richard\\Repository\\ddg2p\\experiments\\mamba_model_def.ckp",
    'max_epochs': 3,
}
