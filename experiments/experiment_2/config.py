# # layer Mamba with UK English only

# Define the data config
data_config = {
    'd_in': 256,  # Byte encoding
    'd_out': 256,  # Byte decoding
    'data_path': 'C:\\Users\\Richard\\Repository\\g2p_data\\ipa-dict\\data\\',
    'csv_name': 'en_data.csv',
    'batch_size': 64,
    'languages': ['en_UK']
}

# Define the network
model_config = {
    'model': 'mamba',
    'd_model': 256,
    'n_layers': 3,
    'PATH': "C:\\Users\\Richard\\Repository\\ddg2p\\experiments\\experiment_2\\mamba_model_en.ckp",
    'max_epochs': 40,
}
