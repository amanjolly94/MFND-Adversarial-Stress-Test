import argparse
import pprint
import numpy as np
import os
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from utils.logger import create_logger
from utils.tokenizer import build_tokenizer
from models import build_model, build_text_model, build_img_model
from dataset import NewsDataset, collate_fn


def parse_args():

    parser = argparse.ArgumentParser("Multi-Modal Training")
    parser.add_argument('--config_path', type=str, default="configs/cnn.yaml", help="Config path")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    os.makedirs(cfg.checkpoints, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")
    logger, tb_dir = create_logger(cfg)
    logger.info(pprint.pformat(cfg))

    writer = SummaryWriter(log_dir=tb_dir)

    tokenizer = build_tokenizer(cfg)
    cfg.model.vocab_size = len(tokenizer)

    trans_fn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = NewsDataset(cfg.data, tokenizer, trans_fn)
    train_length = int(0.8 * len(dataset))
    val_length = len(dataset) - train_length

    # Randomly split the dataset into training and test sets
    train_data, val_data = random_split(dataset, [train_length, val_length])

    train_loader = DataLoader(train_data, batch_size=cfg.data.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_data, batch_size=cfg.data.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)

    # model = BiLSTMVGGClassifier(cfg.model)
    # model = build_text_model(cfg.model)
    model = build_img_model(cfg.model)
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    val_acc_track = 0
    
    for epoch in range(cfg.training.epochs):

        model.train()
        losses = 0

        with tqdm(train_loader) as tloader:
            for batch in tloader:
                tloader.set_description(f"Train Epoch:{epoch}")

                text, img, target = batch
                text, img, target = text.to(device), img.to(device), target.to(device)

                optimizer.zero_grad()
                out = model(text=text, img=img)

                loss = criterion(out.squeeze(), target)
                
                loss.backward()
                optimizer.step()

                losses += loss.item()
                tloader.set_postfix_str(loss.item())
        
        train_loss = losses/len(train_loader)

        model.eval()
        targets = []
        predicts = []
        losses = 0

        with torch.no_grad():
            with tqdm(val_loader) as tloader:
                for batch in tloader:
                    tloader.set_description(f"Val Epoch:{epoch}")

                    text, img, target = batch
                    text, img, target = text.to(device), img.to(device), target.to(device)

                    out = model(text=text, img=img)
                    loss = criterion(out.squeeze(), target)

                    losses += loss.item()

                    targets.append(target.cpu().numpy())
                    tmp = (out.squeeze()>0.5)
                    predicts.append(tmp.to(torch.uint8).cpu().numpy())
                    # predicts.append(torch.tensor(tmp, dtype=torch.uint8).cpu().numpy())

                    tloader.set_postfix_str(loss.item())
            
        val_loss = losses/len(val_loader)
        tar_list_np = np.concatenate(targets)
        pred_list_np = np.concatenate(predicts)

        accuracy = accuracy_score(pred_list_np, tar_list_np)
        precision = precision_score(pred_list_np, tar_list_np)
        f1 = f1_score(pred_list_np, tar_list_np)
        recall = recall_score(pred_list_np, tar_list_np)

        msg = "Epoch: {} Train loss: {} Val loss: {} accuracy: {} precision: {} f1: {}".format(
            epoch, train_loss, val_loss, accuracy, precision, f1, recall
        )

        logger.info(msg)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        writer.add_scalar("Metrics/Accuracy", accuracy, epoch)
        writer.add_scalar("Metrics/Precision", precision, epoch)
        writer.add_scalar("Metrics/Recall", recall, epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)

        # Save weight
        if accuracy > val_acc_track:
            ckpt = {
                "model_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
                "epoch": epoch,
                'accuracy': accuracy,
                'precision' : precision,
                'recall' : recall,
                'f1_score' : f1,
            }

            save_path = os.path.join(cfg.checkpoints, f'{cfg.model.name}_best_model.pt')
            torch.save(ckpt, save_path)
            logger.info(f'Best Model saved with accuracy {accuracy}')
            val_acc_track = accuracy

    logger.info("Training finished")
    msg = "accuracy: {} precision: {} f1: {} recall: {}".format(
        ckpt["accuracy"], ckpt["precision"], ckpt["f1_score"], ckpt["recall"]
    )
    logger.info(msg)

if __name__ == "__main__":
    main()