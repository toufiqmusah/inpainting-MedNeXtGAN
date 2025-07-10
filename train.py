"""train.py"""

import os
import wandb
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from statistics import mean
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LRScheduler

from monai.losses import SSIMLoss
from kornia.losses import PSNRLoss
from generative.losses import PerceptualLoss, PatchAdversarialLoss

from utils import (save_comparison, saving_model)
from config import (LAMBDA_SSIM, LAMBDA_PERCEPT, LAMBDA_L1, LAMBDA_PSNR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScaledPSNRLoss(nn.Module):
    def __init__(self, max_val=2.0, min_db=0.0, max_db=40.0):
        super().__init__()
        self.psnr = PSNRLoss(max_val)
        self.min_db = min_db
        self.max_db = max_db

    def forward(self, pred, target):
        neg_psnr = self.psnr(pred, target)  # negative PSNR
        psnr_db = -neg_psnr
        psnr_norm = (psnr_db - self.min_db) / (self.max_db - self.min_db)
        psnr_norm = torch.clamp(psnr_norm, 0, 1)
        return 1.0 - psnr_norm  # 0 best, 1 worst 


def train_fn(train_dl, G, D,
             criterion_mse, criterion_mae, criterion_perceptual, criterion_ssim, criterion_psnr, criterion_adv,
             optimizer_g, optimizer_d, LRScheduler_G, LRScheduler_D, noise_std=0.05):
    G.train()
    D.train()
    total_loss_g, total_loss_d, total_loss_ssim, total_loss_perceptual, total_loss_psnr = [], [], [], [], []
    
    for i, batch in enumerate(tqdm(train_dl)):
        input_img = batch["mri"].to(device)
        # real_img_clean = batch["label"].to(device)
        #void = batch["mri"].to(device)
        #mask = batch["mask"].to(device)
        real_img_clean = (batch["label"] * batch["mask"]).to(device)
        
        # Generator Forward Pass
        fake_img_clean = G(input_img)

        # Add noise to discriminator inputs only
        real_img_noisy = real_img_clean + noise_std * torch.randn_like(real_img_clean)
        fake_img_noisy = fake_img_clean.detach() + noise_std * torch.randn_like(fake_img_clean)

        # Discriminator Forward Pass
        fake_pred = D(torch.cat([input_img, fake_img_noisy], dim=1))
        if isinstance(fake_pred, list):
            fake_pred = fake_pred[-1]

        real_pred = D(torch.cat([input_img, real_img_noisy], dim=1))
        if isinstance(real_pred, list):
            real_pred = real_pred[-1]

        real_label = torch.rand_like(real_pred) * 0.2 + 0.8 # dynamic label smoothing
        fake_label = torch.rand_like(fake_pred) * 0.2

        # Generator Loss
        #loss_g_gan = criterion_mse(fake_pred, real_label)
        loss_g_gan = criterion_adv(fake_pred, target_is_real=True, for_discriminator=False)
        loss_g_l1 = criterion_mae(fake_img_clean, real_img_clean)
        loss_g_perceptual = criterion_perceptual(fake_img_clean, real_img_clean)
        loss_g_ssim = criterion_ssim(fake_img_clean, real_img_clean)
        loss_g_psnr = criterion_psnr(fake_img_clean, real_img_clean)

        loss_g = loss_g_gan + LAMBDA_L1 * loss_g_l1 + LAMBDA_PERCEPT * loss_g_perceptual + LAMBDA_SSIM * loss_g_ssim + loss_g_psnr * LAMBDA_PSNR

        optimizer_g.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optimizer_g.step()
        LRScheduler_G.step()

        # Discriminator Update
        fake_pred = D(torch.cat([input_img, fake_img_noisy.detach()], dim=1))
        if isinstance(fake_pred, list):
            fake_pred = fake_pred[-1]

        loss_d_fake = criterion_adv(fake_pred, target_is_real=False, for_discriminator=True)
        loss_d_real = criterion_adv(real_pred, target_is_real=True, for_discriminator=True)

        loss_d = (loss_d_real + loss_d_fake) * 0.85

        optimizer_d.zero_grad()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
        optimizer_d.step()
        LRScheduler_D.step()

        # Track gradients
        g_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in G.parameters() if p.grad is not None]))
        d_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in D.parameters() if p.grad is not None]))

        total_loss_g.append(loss_g.item())
        total_loss_d.append(loss_d.item())
        total_loss_ssim.append(loss_g_ssim.item())
        total_loss_psnr.append(loss_g_psnr.item())
        total_loss_perceptual.append(loss_g_perceptual.item())

    return (mean(total_loss_g), mean(total_loss_d), mean(total_loss_ssim), mean(total_loss_perceptual), mean(total_loss_psnr), 
            fake_img_clean.detach().cpu(), real_img_clean.detach().cpu(), input_img.detach().cpu(),
            g_grad_norm.item(), d_grad_norm.item())

def train_loop(train_dl, G, D, num_epoch, LOG_DIV=5, betas=(0.5, 0.999)):
    G.to(device)
    D.to(device)

    optimizer_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=betas)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=1e-5, betas=betas)

    final_lr_factor = 0.15
    LRScheduler_G = LRScheduler.LinearLR(optimizer_g, start_factor=1.0, end_factor=final_lr_factor, total_iters=num_epoch * len(train_dl))
    LRScheduler_D = LRScheduler.LinearLR(optimizer_d, start_factor=1.0, end_factor=final_lr_factor, total_iters=num_epoch * len(train_dl))

    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    criterion_perceptual = PerceptualLoss(spatial_dims=3, is_fake_3d=True, pretrained=True, network_type="squeeze").to(device) # alternatives: "resnet18", "resnet34"
    criterion_ssim = SSIMLoss(k1=0.01, k2=0.03, win_size=11, spatial_dims=3, data_range=2.0, kernel_sigma=1.5, reduction="mean", kernel_type="gaussian",).to(device)
    criterion_adv = PatchAdversarialLoss(criterion = "least_squares").to(device)
    criterion_psnr = ScaledPSNRLoss(max_val=2.0, min_db=0.0, max_db=40.0).to(device)
    # criterion_psnr = PSNRLoss(max_val=2.0).to(device)
    
    total_loss_d, total_loss_g = [], []
    result = {"G": [], "D": []}

    for e in range(num_epoch): 
        print(f"\nEpoch {e+1}/{num_epoch}")

        noise_std = max(0.05 * (1 - e / num_epoch), 0.01)
        loss_g, loss_d, loss_ssim, loss_perceptual, loss_psnr, fake_img, real_img, input_img, g_grad_norm, d_grad_norm = train_fn(
                                                                                                                    train_dl, G, D, criterion_mse, 
                                                                                                                    criterion_mae, criterion_perceptual, criterion_ssim, criterion_psnr, criterion_adv,
                                                                                                                    optimizer_g, optimizer_d, LRScheduler_G, LRScheduler_D, noise_std)
    
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']

        total_loss_d.append(loss_d)
        total_loss_g.append(loss_g)
        result["G"].append(loss_g)
        result["D"].append(loss_d)

        print(f"Generator Loss: {loss_g:.4f}, Discriminator Loss: {loss_d:.4f}")
        print(f"Learning rates: G: {current_lr_g:.6f}, D: {current_lr_d:.6f}")


        wandb.log({
            "Generator Loss": loss_g,
            "Discriminator Loss": loss_d,
            "G Grad Norm": g_grad_norm,
            "D Grad Norm": d_grad_norm,
            "SSIM Loss": loss_ssim,
            "Perceptual Loss": loss_perceptual,
            "PSNR Loss": loss_psnr,
            "epoch": e + 1
        })

        if (e + 1) % LOG_DIV == 0:
            saving_model(D, G, e + 1)
            save_comparison(real_img, fake_img, input_img, e + 1)

    print("Training completed successfully")

    return G, D