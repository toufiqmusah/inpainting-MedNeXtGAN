"""inference.py, & resampling"""

import os
import torch
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from config import DEVICE
from models import MedNextGenerator3D

from monai.transforms import (
    Compose,
    Spacing,
    DivisiblePad,
    CenterSpatialCrop,
    ScaleIntensityRangePercentiles,
    ResampleToMatch
)
from monai.data import MetaTensor

def argument_parser():
    parser = argparse.ArgumentParser(description="Inference script for BraTS-InPainting")
    parser.add_argument("--model_weights", type=str, required=True, help="Model Weights Path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for inpainted results")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing voided MRI files")
    return parser.parse_args()

void_transforms = Compose([
    Spacing(pixdim=(2, 2, 2), mode="bilinear"),
    CenterSpatialCrop(roi_size=(128, 128, 80)),
    DivisiblePad(k=16, mode="constant", constant_values=0),
    ScaleIntensityRangePercentiles(
        lower=0,
        upper=99.5,
        b_min=-1.0,
        b_max=1.0
    ),
])

mask_transforms = Compose([
    Spacing(pixdim=(2, 2, 2), mode="nearest"),
    CenterSpatialCrop(roi_size=(128, 128, 80)),
    DivisiblePad(k=16, mode="constant", constant_values=0),
])

def read_paths_pair(root_dir: str):
    void_files = sorted(glob(os.path.join(root_dir, "**", "*t1n-voided.nii.gz"), recursive=True))
    mask_files = sorted(glob(os.path.join(root_dir, "**", "*mask.nii.gz"), recursive=True))
    return void_files, mask_files

def run_inference(model, input_dir, output_dir, void_transforms, mask_transforms):
    os.makedirs(output_dir, exist_ok=True)

    resampler = ResampleToMatch(mode="bilinear", padding_mode="border", align_corners=False)
    void_paths, mask_paths = read_paths_pair(input_dir)
    
    for void_path, mask_path in zip(void_paths, mask_paths):
        v = nib.load(void_path)
        m = nib.load(mask_path)
        
        v_f = np.expand_dims(v.get_fdata(), axis=0)
        m_f = np.expand_dims(m.get_fdata(), axis=0)
        
        v_original_meta = MetaTensor(
            torch.from_numpy(v_f).float(),
            affine=torch.from_numpy(v.affine).float(),
            meta={'spatial_shape': v_f.shape[1:]}
        )
        
        v_meta = MetaTensor(
            torch.from_numpy(v_f).float(),
            affine=torch.from_numpy(v.affine).float(),
            meta={'spatial_shape': v_f.shape[1:]}
        )
        
        m_meta = MetaTensor(
            torch.from_numpy(m_f).float(),
            affine=torch.from_numpy(m.affine).float(),
            meta={'spatial_shape': m_f.shape[1:]}
        )
        
        v_transformed = void_transforms(v_meta)  # Shape: (1, 128, 128, 80)
        m_transformed = mask_transforms(m_meta)  # Shape: (1, 128, 128, 80)
        
        v_transformed_np = v_transformed.detach().cpu().numpy()
        m_transformed_np = m_transformed.detach().cpu().numpy()
        
        subject_channels = np.concatenate([v_transformed_np, m_transformed_np], axis=0)  
        subject_batch = np.expand_dims(subject_channels, axis=0)  # New shape: (1, 2, 128, 128, 80)
        
        print(f"Processing {os.path.basename(void_path)}, with shape {subject_batch.shape})")
        
        input_tensor = torch.from_numpy(subject_batch).float()
        output = model(input_tensor)
        
        print(f"Output shape: {output.shape}")
        
        output_squeezed = output.squeeze(0)  
        if output_squeezed.dim() == 4:  
            output_squeezed = output_squeezed[0:1] 
        
        output_meta = MetaTensor(
            output_squeezed,
            affine=v_transformed.affine, 
            meta=v_transformed.meta
        )
        
        # Resample output to match original input "v"
        output_resampled = resampler(output_meta, v_original_meta)
        
        print(f"Resampled output shape: {output_resampled.shape}")
        output_resampled_np = output_resampled.detach().cpu().numpy()
        
        if output_resampled_np.ndim == 4:
            output_resampled_np = output_resampled_np[0]
        
        output_nib = nib.Nifti1Image(output_resampled_np, affine=v.affine, header=v.header.copy())
        output_filename = os.path.join(output_dir, os.path.basename(void_path).replace("-t1n-voided", ""))
        nib.save(output_nib, output_filename)
        
        print(f"Saved inpainted image to {output_filename}, shape: {output_nib.get_fdata().shape}")

if __name__ == "__main__":
    args = argument_parser()
    net = MedNextGenerator3D().to(DEVICE)
    net.load_state_dict(torch.load(args.model_weights, map_location=DEVICE))
    net.eval()
    
    run_inference(net, args.input_dir, args.output_dir, void_transforms, mask_transforms)