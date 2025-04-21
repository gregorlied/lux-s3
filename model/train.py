import os
import math
import shutil
import pickle
import zipfile 
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup
from torchvision import transforms
from tqdm.auto import tqdm

from dataset import LuxDataset, LuxFlipTransform, LuxSapDataset
from loss import DiceLoss
from model import UNet
from utils import Accuracy, seed_everything, load_config

def get_samples(path):
    
    with open(path, 'rb') as fp:
        samples = pickle.load(fp)
    df = pd.DataFrame(samples)

    episodes = [episode for episode, _ in df.groupby("episode")]
    _, test_idx = train_test_split(range(len(episodes)), test_size=0.2, random_state=42)
    test_episodes = [episodes[i] for i in test_idx]

    train_samples = []
    test_samples = []
    for episode, group in df.groupby("episode"):
        del group["episode"]
        samples = zip(*group.values.T)
        if episode in test_episodes:
            test_samples.extend(samples)
        else:
            train_samples.extend(samples)

    return train_samples, test_samples


def train(model, train_dataloader, test_dataloader, criterion, optimizer, lr_scheduler, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    accuracy = Accuracy(model.num_agents, model.num_classes)

    for epoch in range(num_epochs):
        model.train()
        model = model.to(device)

        epoch_loss = 0.0
        accuracy.reset()
        for states, agent_ids, actions, masks in tqdm(train_dataloader, leave=False):
            states = states.to(device).float()
            agent_ids = agent_ids.to(device).long()
            actions = actions.to(device).long()
            masks = masks.to(device).long()

            optimizer.zero_grad()

            policy = model(states, masks, agent_ids)

            if isinstance(criterion, nn.CrossEntropyLoss):
                actions[masks == 0] = -100
                loss = criterion(policy, actions)
            else:
                loss = criterion(policy, actions, masks)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += loss.item() * len(policy)
            accuracy.update(policy, actions, masks, agent_ids)

        print("================== TRAIN ==================")
        epoch_loss = epoch_loss / len(test_dataloader.dataset)
        last_lr = lr_scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | LR: {last_lr:.6f}")
        accuracy.print()

        model.eval()
        epoch_loss = 0.0
        accuracy.reset()
        for states, agent_ids, actions, masks in tqdm(test_dataloader, leave=False):
            states = states.to(device).float()
            agent_ids = agent_ids.to(device).long()
            actions = actions.to(device).long()
            masks = masks.to(device).long()

            with torch.no_grad():
                policy = model(states, masks, agent_ids)

            if isinstance(criterion, nn.CrossEntropyLoss):
                actions[masks == 0] = -100
                loss = criterion(policy, actions)
            else:
                loss = criterion(policy, actions, masks)

            epoch_loss += loss.item() * len(policy)
            accuracy.update(policy, actions, masks, agent_ids)

        print("================== TEST ==================")
        epoch_loss = epoch_loss / len(test_dataloader.dataset)
        last_lr = lr_scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | LR: {last_lr:.6f}")
        accuracy.print()

    model.eval()
    model.cpu()
    x = torch.rand(1, model.num_channels + model.num_global_channels, 24, 24)
    mask = torch.ones((1, 24, 24)).long()
    agent_ids = torch.rand(1).long()
    traced = torch.jit.trace(model, (x, mask, agent_ids))
    traced.save(f'model.pth')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default= "./configs/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    temp_dir = config["data"]["temp_dir"]
    data_dir = config["data"]["data_dir"]
    sample_file = config["data"]["sample_file"]
    sub_ids = config["data"]["sub_ids"]
    use_sap_dataset_bool = config["data"]["use_sap_dataset_bool"]

    num_channels = config["model"]["num_channels"]
    num_global_channels = config["model"]["num_global_channels"]
    num_classes = config["model"]["num_classes"]
    num_agents = config["model"]["num_agents"]
    num_blocks = config["model"]["num_blocks"]
    dit_head_bool = config["model"]["dit_head_bool"]
    flip_transform_bool = config["model"]["flip_transform_bool"]
    use_weighted_dice_bool = config["model"]["use_weighted_dice_bool"]

    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    seed = config["training"]["seed"]
    warmup_ratio = config["training"]["warmup_ratio"]

    seed_everything(seed)

    sample_path = os.path.join(data_dir, sample_file)
    train_samples, test_samples = get_samples(sample_path)

    os.mkdir(temp_dir)
    for sub in sub_ids:
        path = os.path.join(data_dir, str(sub) + '.zip')
        with zipfile.ZipFile(path) as z:
            z.extractall(temp_dir)

    model = UNet(
        num_channels=num_channels,
        num_global_channels=num_global_channels,
        num_classes=num_classes,
        num_agents=num_agents,
        num_blocks=num_blocks,
        dit_head_bool=dit_head_bool,
    )

    if flip_transform_bool:
        transform = transforms.Compose([LuxFlipTransform()])
    else:
        transform = None

    if use_weighted_dice_bool:
        criterion = DiceLoss(num_classes, weights=[1., 50.])
    else:
        criterion = nn.CrossEntropyLoss()

    if use_sap_dataset_bool:
        train_dataset = LuxSapDataset(temp_dir, train_samples, transform=transform)
        test_dataset = LuxSapDataset(temp_dir, test_samples)
    else:
        train_dataset = LuxDataset(temp_dir, train_samples, transform=transform)
        test_dataset = LuxDataset(temp_dir, test_samples)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = math.ceil(num_epochs * num_update_steps_per_epoch)
    num_warmup_steps = math.ceil(num_training_steps * warmup_ratio)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    train(
        model,
        train_dataloader,
        test_dataloader,
        criterion, 
        optimizer,
        lr_scheduler,
        num_epochs=num_epochs
    )

    shutil.rmtree(temp_dir)
