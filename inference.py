"""inference.py - Using NiBabel for resampling"""

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
    EnsureTyped,
    DivisiblePad,
    CenterSpatialCrop,
    EnsureChannelFirst,
    ScaleIntensityRangePercentiles)


def argument_parser():
    parser = argparse.ArgumentParser(description="Inference script for BraTS-InPainting")
    parser.add_argument("--model_weights", type=str, required=True, help="Model Weights Path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for inpainted results")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing voided MRI files")

    return parser.parse_args()


def read_paths_pair(root_dir: str):
    void_mri_files = sorted(glob(os.path.join(root_dir, "**", "*t1n-voided.nii.gz"), recursive=True))
    mask_files = sorted(glob(os.path.join(root_dir, "**", "*mask.nii.gz"), recursive=True))

    return void_mri_files, mask_files

_pad = DivisiblePad(k=8, mode="constant", constant_values=0)         
transforms = Compose([
    EnsureChannelFirst(),
    Spacing(pixdim=(2, 2, 2), mode=("bilinear")),
    _pad,
    ScaleIntensityRangePercentiles(
        lower=0,
        upper=99.5,
        b_min=-1.0,
        b_max=1.0
    )
])

def make_postproc(orig_depth):
    """Crop pad and center-crop to 256×256×155."""
    before = _pad.pad_width[2][0]              
    return Compose([lambda x: x[..., before:before + orig_depth], CenterSpatialCrop((256, 256, 155))])

def load_case(void_path, mask_path):

    void = nib.load(void_path)
    mask = nib.load(mask_path)
    void_arr = void.get_fdata()
    mask_arr = mask.get_fdata()
    pair = np.stack([void_arr, mask_arr], axis=0) 

    return pair, void.affine, void.header.copy()

def run_inference(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    void_paths, mask_paths = read_paths_pair(input_dir)
    
    for void_path, mask_path in zip(void_paths, mask_paths):
        pair, affine, header = load_case(void_path, mask_path)
        pair = transforms(pair)

        input_tensor = torch.tensor(pair, dtype=torch.float32).unsqueeze(0).to(DEVICE)  
        model.eval()
        
        with torch.no_grad():
            output_tensor = model(input_tensor)

        postproc = make_postproc(pair.shape[-1])
        output_tensor = postproc(output_tensor.cpu().squeeze(0).numpy())

        output_img = nib.Nifti1Image(output_tensor, affine=affine, header=header)
        output_filename = os.path.join(output_dir, os.path.basename(void_path).replace("t1n-voided", ""))
        nib.save(output_img, output_filename)
        print(f"Saved inpainted image to {output_filename}")

if __name__ == "__main__":
    args   = argument_parser()
    net    = MedNextGenerator3D().to(DEVICE)
    net.load_state_dict(torch.load(args.model_weights, map_location=DEVICE))
    net.eval()
    run_inference(net, args.input_dir, args.output_dir)