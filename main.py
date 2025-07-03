"""main.py"""

import sys
import wandb
import torch
import argparse

from train import train_loop
from data import get_dataloader
from models import (MedNextGenerator3D, PatchDiscriminator3D)
from config import (NUM_EPOCHS, INPUT_DIR, BATCH_SIZE, WANDB_PROJECT)

def parse_args():
    parser = argparse.ArgumentParser(description="Train MedNextGAN for MRI inpainting")
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR, help="Path to the input data directory")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs for training")
    return parser.parse_args()

args = parse_args()

# wandb login
WANDB_API_KEY = "8b67af0ea5e8251ee45c6180b5132d513b68c079"  
wandb.login(key=WANDB_API_KEY)

dataloader = get_dataloader(args.input_dir, batch_size=args.batch_size)

# model instances
G = MedNextGenerator3D(input_channels=2, output_channels=1)
D = PatchDiscriminator3D

wandb.init(project=WANDB_PROJECT)

trained_G, trained_D = train_loop(dataloader, G, D, args.num_epochs)
