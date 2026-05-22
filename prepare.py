import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

input_file_path = os.path.join(DATA_DIR, 'tinyshakespeare.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    print(f"Downloading {data_url} to {input_file_path}")
    urllib.request.urlretrieve(data_url, input_file_path)
    print("Download complete.")
else:
    print(f"File {input_file_path} already exists.")
