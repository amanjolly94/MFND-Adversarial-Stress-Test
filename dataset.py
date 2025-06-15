import pandas as pd
import torch
from PIL import Image
from urllib.request import urlopen
import requests
from io import BytesIO
import ast
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NewsURLDataset(Dataset):

    def __init__(self, cfg, tokenizer, image_transform):

        # self.data = self._clean_data(data_path, txt_col)
        self.data = pd.read_csv(cfg.data_path)
        self.txt_col = cfg.txt_col
        self.img_col = cfg.img_col
        self.tar_col = cfg.tar_col
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def _clean_data(self, data_path, txt_col):

        def _decode_if_bytes(value):
            if value.startswith("b'") or value.startswith('b"'):
                value = ast.literal_eval(value).decode('utf-8', errors='replace')
            return value
        
        df = pd.read_csv(data_path, encoding= 'latin1')
        df[txt_col] = df[txt_col].apply(_decode_if_bytes)

        return df
    
    def _read_image_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:  # Check if the request was successful
            img_data = BytesIO(response.content)
            img = Image.open(img_data)
            return img
        else:
            return None  # Or raise an exception

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        text = sample[self.txt_col]
        image_url = sample[self.img_col]
        label = sample[self.tar_col]

        text = self.tokenizer.encode(
            text, 
            add_special_tokens=True,
            return_tensors="pt"
        )

        # read img_url with PIL and convert to RGB
        img = self._read_image_from_url(image_url).convert('RGB')
        # img = Image.open(urlopen(image_url)).convert('RGB')
        img = self.image_transform(img)

        label = torch.tensor(label, dtype=torch.float)

        return text, img, label


class NewsDataset(Dataset):

    def __init__(self, cfg, tokenizer, image_transform):

        # self.data = self._clean_data(data_path, txt_col)
        self.data = pd.read_csv(cfg.data_path)
        self.txt_col = cfg.txt_col
        self.img_col = cfg.img_col
        self.tar_col = cfg.tar_col
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        text = sample[self.txt_col]
        image_path = sample[self.img_col]
        label = sample[self.tar_col]

        text = self.tokenizer.encode(
            text, 
            add_special_tokens=True,
            return_tensors="pt"
        )

        # read img_url with PIL and convert to RGB
        img = Image.open(image_path).convert('RGB')
        img = self.image_transform(img)

        label = torch.tensor(label, dtype=torch.float)

        return text, img, label

class FilteredNewsDataset(Dataset):

    def __init__(self, cfg, tokenizer, image_transform):
        self.data = pd.read_csv(cfg.data_path)
        self.txt_col = cfg.txt_col
        self.idx_col = cfg.idx_col
        self.img_root = cfg.img_root
        self.tar_col = cfg.tar_col
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        text = sample[self.txt_col]
        image_path = f"{self.img_root}/{sample[self.idx_col]}.jpg"
        label = sample[self.tar_col]

        text = self.tokenizer.encode(
            text, 
            add_special_tokens=True,
            return_tensors="pt"
        )

        # read img_url with PIL and convert to RGB
        try:
            img = Image.open(image_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))

        img = self.image_transform(img)

        label = torch.tensor(label, dtype=torch.float)

        return text, img, label

def collate_fn(batch):
    # Separate the text, images, and labels from the batch
    text_list, img_list, label_list = [], [], []

    for (_text, _img, _label) in batch:
        text_list.append(_text[0])
        img_list.append(_img)
        label_list.append(_label)

    # Pad the sequences in text_list so they all have the same length
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)

    # Stack all the images and labels into a single tensor
    img_list = torch.stack(img_list)
    label_list = torch.stack(label_list)

    return text_list, img_list, label_list

