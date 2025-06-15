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
from models import build_model
from dataset import FilteredNewsDataset, collate_fn

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser("Multi-Modal Training")
    parser.add_argument('--config_path', type=str, default="configs/bert.yaml", help="Config path")
    args = parser.parse_args()
    return args

def main():
    # Parse command line arguments
    args = parse_args()
    # Load configuration from yaml file
    cfg = OmegaConf.load(args.config_path)

    # Create directory for saving model weights
    weight_dir = f"{cfg.checkpoints}/{cfg.model.modality}/{cfg.model.name}/{cfg.data.name}"
    os.makedirs(weight_dir, exist_ok=True)

    # Check if CUDA is available and set device accordingly
    if torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    # Create logger and TensorBoard writer
    logger, tb_dir = create_logger(cfg)
    logger.info(pprint.pformat(cfg))
    writer = SummaryWriter(log_dir=tb_dir)

    # Build tokenizer and update vocabulary size in configuration
    tokenizer = build_tokenizer(cfg)
    cfg.model.vocab_size = len(tokenizer)

    # Define image transformations
    trans_fn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and split it into training and validation sets
    dataset = FilteredNewsDataset(cfg.data, tokenizer, trans_fn)
    train_length = int(0.8 * len(dataset))
    val_length = len(dataset) - train_length
    train_data, val_data = random_split(dataset, [train_length, val_length])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_data, batch_size=cfg.data.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_data, batch_size=cfg.data.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)

    # Build model and move it to the device
    model = build_model(cfg.model)
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    val_acc_track = 0

    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        losses = 0

        # Training step
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

        # Validation step
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

                    tloader.set_postfix_str(loss.item())
            
        val_loss = losses/len(val_loader)
        tar_list_np = np.concatenate(targets)
        pred_list_np = np.concatenate(predicts)

        # Calculate metrics
        accuracy = accuracy_score(pred_list_np, tar_list_np)
        precision = precision_score(pred_list_np, tar_list_np)
        f1 = f1_score(pred_list_np, tar_list_np)
        recall = recall_score(pred_list_np, tar_list_np)

        # Log metrics
        msg = "Epoch: {} Train loss: {} Val loss: {} accuracy: {} precision: {} f1: {}".format(
            epoch, train_loss, val_loss, accuracy, precision, f1, recall
        )
        logger.info(msg)

        # Write metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Metrics/Accuracy", accuracy, epoch)
        writer.add_scalar("Metrics/Precision", precision, epoch)
        writer.add_scalar("Metrics/Recall", recall, epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)

        # Save best model
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

            save_path = f"{weight_dir}/best_model.pt"
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