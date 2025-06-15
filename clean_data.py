import pandas as pd
from PIL import Image
from urllib.request import urlopen
import ast
from tqdm import tqdm
import numpy as np
import requests
from io import BytesIO

df = pd.read_csv("data/D.csv", encoding= 'latin1')

def decode_if_bytes(value):
    if value.startswith("b'") or value.startswith('b"'):
        value = ast.literal_eval(value).decode('utf-8', errors='replace')
    return value

df['Text'] = df['Text'].apply(decode_if_bytes)

rows_list = []

def read_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:  # Check if the request was successful
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        return img
    else:
        return None  # Or raise an exception

# for idx, row in tqdm(df.iterrows()):
#     url = row['Image']

#     try:
#         # img = Image.open(urlopen(url)).convert('RGB')
#         img = read_image_from_url(url)
#         img = img.convert("RGB")
#         rows_list.append(row)
#     except Exception as e:
#         print(url)
#         continue

for idx, row in tqdm(df.iterrows()):
    url = row['Image']

    try:
        # img = Image.open(urlopen(url)).convert('RGB')
        img = read_image_from_url(url)
        img = img.convert("RGB")
        
        local_path = f"data/D_images/image_{idx}.jpg"
        img.save(local_path)
        
        row["local_url"] = local_path
        rows_list.append(row)
    except Exception as e:
        print(url)
        continue

revised_df = pd.DataFrame(rows_list)

print(df.shape, revised_df.shape)

revised_df.to_csv('data/D_revised_03.csv', index=False)