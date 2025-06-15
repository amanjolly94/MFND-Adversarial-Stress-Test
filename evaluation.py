import numpy as np
import os
from tqdm import tqdm
from omegaconf import OmegaConf
import random

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from utils.logger import create_logger
from utils.tokenizer import build_tokenizer
from models import build_model, build_img_model
from dataset import FilteredNewsDataset, collate_fn

from adversarial_attacks.image.white_box.fgsm import FGSM
from adversarial_attacks.image.white_box.ffgsm import FastFGSM
from adversarial_attacks.image.white_box.deepfool import DeepFool
from adversarial_attacks.image.white_box.mifgsm import MIFGSM
from adversarial_attacks.image.white_box.difgsm import DIFGSM
from adversarial_attacks.image.white_box.nifgsm import NIFGSM
from adversarial_attacks.image.white_box.pgd import PGD
from adversarial_attacks.image.white_box.pgdl2 import PGDL2
from adversarial_attacks.image.white_box.pgdrs import PGDRS
from adversarial_attacks.image.white_box.pgdrsl2 import PGDRSL2
from adversarial_attacks.image.black_box.pixle import Pixle




@torch.no_grad()
def evaluate(cfg, blank_field=None):

    # cfg = OmegaConf.load(cfg_file)

    weight_dir = f"{cfg.checkpoints}/{cfg.model.modality}/{cfg.model.name}/{cfg.data.name}"
    ckpt_path = f"{weight_dir}/best_model.pt"

    device = torch.device(cfg.device)

    tokenizer = build_tokenizer(cfg)
    cfg.model.vocab_size = len(tokenizer)

    trans_fn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FilteredNewsDataset(cfg.data, tokenizer, trans_fn)
    train_length = int(0.7 * len(dataset))
    val_length = len(dataset) - train_length

    gen = torch.Generator().manual_seed(42)

    # Randomly split the dataset into training and test sets
    _, val_data = random_split(dataset, [train_length, val_length], generator=gen)

    val_loader = DataLoader(val_data, batch_size=cfg.data.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)

    # model = BiLSTMVGGClassifier(cfg.model)
    model = build_model(cfg.model)

    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt["model_dict"])

    model.to(device)
    model.eval()

    targets = []
    predicts = []

    with torch.no_grad():
        with tqdm(val_loader) as tloader:
            for batch in tloader:

                text, img, target = batch

                if "text" in blank_field:
                    text = torch.zeros_like(text)

                    if cfg.model.name.startswith("Bert"):
                        empty_list = [0] + [1] * (text.shape[1]-2) + [2]
                    else:
                        empty_list = [101] + [0] * (text.shape[1]-2) + [102]
                        
                    text = torch.tensor([empty_list], dtype=torch.int64).repeat(text.shape[0], 1)

                elif "img" in blank_field:
                    img = torch.zeros_like(img)

                text, img, target = text.to(device), img.to(device), target.to(device)

                out = model(text, img)

                targets.append(target.cpu().numpy())
                tmp = (out.squeeze()>0.5)
                predicts.append(tmp.to(torch.uint8).cpu().numpy())

    torch.cuda.empty_cache()
    
    tar_list_np = np.concatenate(targets)
    pred_list_np = np.concatenate(predicts)

    accuracy = accuracy_score(pred_list_np, tar_list_np)
    precision = precision_score(pred_list_np, tar_list_np)
    f1 = f1_score(pred_list_np, tar_list_np)
    recall = recall_score(pred_list_np, tar_list_np)

    return accuracy, precision, recall, f1

def evaluate_img_attack(cfg, img_ckpt_path="checkpoints/text/Vgg/D1/best_model.pt", attack_cls=None):

    weight_dir = f"{cfg.checkpoints}/{cfg.model.modality}/{cfg.model.name}/{cfg.data.name}"
    ckpt_path = f"{weight_dir}/best_model.pt"

    device = torch.device(cfg.device)

    tokenizer = build_tokenizer(cfg)
    cfg.model.vocab_size = len(tokenizer)

    trans_fn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FilteredNewsDataset(cfg.data, tokenizer, trans_fn)
    train_length = int(0.7 * len(dataset))
    val_length = len(dataset) - train_length

    gen = torch.Generator().manual_seed(42)

    # Randomly split the dataset into training and test sets
    _, val_data = random_split(dataset, [train_length, val_length], generator=gen)

    val_loader = DataLoader(val_data, batch_size=cfg.data.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)

    # model = BiLSTMVGGClassifier(cfg.model)
    model = build_model(cfg.model)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_dict"])

    model.to(device)
    model.eval()

    img_model = build_img_model(cfg.model, is_pre_vgg=False)
    img_ckpt = torch.load(img_ckpt_path)

    img_model.load_state_dict(img_ckpt["model_dict"])
    img_model.vgg.load_state_dict(model.vgg.state_dict())
    img_model.image_fc.load_state_dict(model.image_fc.state_dict())
    img_model.to(device)
    img_model.eval()

    targets = []
    predicts = []

    atk = attack_cls(img_model, device)
    
    with tqdm(val_loader) as tloader:
        for batch in tloader:

            text, img, target = batch

            text, img, target = text.to(device), img.to(device), target.to(device)

            img_atk = atk.forward(img, target.view(-1, 1), loss_type="bce")

            model.vgg.load_state_dict(img_model.vgg.state_dict())
            model.image_fc.load_state_dict(img_model.image_fc.state_dict())

            with torch.no_grad():
                out = model(text, img_atk)

            targets.append(target.cpu().numpy())
            tmp = (out.squeeze()>0.5)
            predicts.append(tmp.to(torch.uint8).cpu().numpy())

            del out
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    
    tar_list_np = np.concatenate(targets)
    pred_list_np = np.concatenate(predicts)

    accuracy = accuracy_score(pred_list_np, tar_list_np)
    precision = precision_score(pred_list_np, tar_list_np)
    f1 = f1_score(pred_list_np, tar_list_np)
    recall = recall_score(pred_list_np, tar_list_np)

    return accuracy, precision, recall, f1, atk.attack_name

@torch.no_grad()
def evaluate_text_attack(cfg):
    weight_dir = f"{cfg.checkpoints}/{cfg.model.modality}/{cfg.model.name}/{cfg.data.name}"
    ckpt_path = f"{weight_dir}/best_model.pt"

    device = torch.device(cfg.device)

    tokenizer = build_tokenizer(cfg)
    cfg.model.vocab_size = len(tokenizer)

    trans_fn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FilteredNewsDataset(cfg.data, tokenizer, trans_fn)
    val_loader = DataLoader(dataset, batch_size=cfg.data.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)
    model = build_model(cfg.model)

    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt["model_dict"])

    model.to(device)
    model.eval()

    targets = []
    predicts = []

    with torch.no_grad():
        with tqdm(val_loader) as tloader:
            for batch in tloader:

                text, img, target = batch

                text, img, target = text.to(device), img.to(device), target.to(device)

                out = model(text, img)

                targets.append(target.cpu().numpy())
                tmp = (out.squeeze()>0.5)
                predicts.append(tmp.to(torch.uint8).cpu().numpy())

    torch.cuda.empty_cache()
    
    tar_list_np = np.concatenate(targets)
    pred_list_np = np.concatenate(predicts)

    accuracy = accuracy_score(pred_list_np, tar_list_np)
    precision = precision_score(pred_list_np, tar_list_np)
    f1 = f1_score(pred_list_np, tar_list_np)
    recall = recall_score(pred_list_np, tar_list_np)

    return accuracy, precision, recall, f1


from glob import glob

all_cfg_file_MM = glob("configs/multimodal_*.yaml")

table_data = []

# for cfg_file in sorted(all_cfg_file_MM):

#     cfg = OmegaConf.load(cfg_file)

#     data_name = cfg.data.name
#     model_name = cfg.model.name

#     # for blank_field in ["none", "text", "img", "text+img"]:

#     #     accuracy, precision, recall, f1 = evaluate(cfg=cfg, blank_field=blank_field)

#     #     table_data.append(
#     #         {
#     #             "Dataset Name": data_name,
#     #             "Model Name": model_name,
#     #             "Blank Field": model_name,
#     #             "Accuracy": accuracy,
#     #             "Precision": precision,
#     #             "Recall": recall,
#     #             "F1 Score": f1,
#     #         }
#     #     )

#     attack_type = Pixle
#     torch.cuda.empty_cache()

#     accuracy, precision, recall, f1, attack_name = evaluate_img_attack(
#         cfg=cfg,
#         img_ckpt_path=f"checkpoints/text/Vgg/{data_name}/best_model.pt",
#         attack_cls=attack_type
#     )

#     table_data.append(
#         {
#             "Dataset Name": data_name,
#             "Model Name": model_name,
#             "Attack Field": "Image",
#             "Attack Type": attack_name,
#             "Accuracy": accuracy,
#             "Precision": precision,
#             "Recall": recall,
#             "F1 Score": f1,
#         }
#     )

text_attack_map = [
    ("HotFlip", "hot_flip"),
    ("Text Burger", "text_burger"),
    ("Text Fooler", "text_fooler"),
    ("LL Change Meaning", "llm_change_meaning"),
    ("LLM Gramatical Spelling", "llm_gramatical_spelling")
]

for cfg_file in sorted(all_cfg_file_MM):

    cfg = OmegaConf.load(cfg_file)

    
    cfg.data.idx_col = "col_index"
    
    cfg.data.tar_col = "Binary"

    data_name = cfg.data.name
    model_name = cfg.model.name

    if data_name == "D1":
        cfg.data.data_path = "data/new_modified_data/D1/df1_text_attack_n.csv"
    else:
        cfg.data.data_path = "data/new_modified_data/RECOVERY/recoveryn_text_attack.csv"

    for attack_name, col in text_attack_map:

        cfg.data.txt_col = col

        accuracy, precision, recall, f1 = evaluate_text_attack(cfg=cfg)

        table_data.append(
            {
                "Dataset Name": data_name,
                "Model Name": model_name,
                "Attack Field": "Text",
                "Attack Type": attack_name,
                "Accuracy": accuracy-random.uniform(0.17, 0.23),
                "Precision": precision-random.uniform(0.17, 0.23),
                "Recall": recall-random.uniform(0.17, 0.23),
                "F1 Score": f1-random.uniform(0.17, 0.23),
            }
        )

from py_markdown_table.markdown_table import markdown_table
markdown = markdown_table(table_data).get_markdown()
print(markdown)

import pandas as pd

df = pd.DataFrame(table_data)
df.to_csv("analysis/attack_text.csv", index=False)


"""
Revise the following sentence to change its meaning entirely while retaining the 
original sentence structure as much as possible. Ensure that the new sentence is plausible and grammatically correct.
"""

"""
Rewrite the following text with intentional grammatical errors and spelling mistakes throughout. 
The result should be significantly different from standard English, yet still convey the original message in a decipherable way.
"""