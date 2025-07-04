"""inference.py - Using NiBabel for resampling"""

import os
import torch
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from config import DEVICE
from models import MedNextGenerator3D
from nibabel.processing import resample_from_to

from monai.transforms import (
    Compose,
    Spacing,
    DivisiblePad,
    CenterSpatialCrop,
    ScaleIntensityRangePercentiles)

from monai.data import Dataset, DataLoader

def argument_parser():
    parser = argparse.ArgumentParser(description="Inference script for BraTS-InPainting")
    parser.add_argument("--model_weights", type=str, required=True, help="Model Weights Path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for inpainted results")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing voided MRI files")

    return parser.parse_args()

void_transforms = Compose([
    Spacing(pixdim=(2, 2, 2)),
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
    # Note: No intensity scaling for mask
])

def read_paths_pair(root_dir: str):
    void_files = sorted(glob(os.path.join(root_dir, "**", "*t1n-voided.nii.gz"), recursive=True))
    mask_files = sorted(glob(os.path.join(root_dir, "**", "*mask.nii.gz"), recursive=True))

    return void_files, mask_files


def run_inference(model, input_dir, output_dir, void_transforms, mask_transforms):
    os.makedirs(output_dir, exist_ok=True)

    void_paths, mask_paths = read_paths_pair(input_dir)

    for void_path, mask_path in zip(void_paths, mask_paths):
        v = np.expand_dims(nib.load(void_path), axis=0)
        m = np.expand_dims(nib.load(mask_path), axis=0)

        v_transformed = void_transforms(v)
        m_transformed = mask_transforms(m)

        subject = np.stack([v_transformed, m_transformed], axis=0)

        print(f"Processing {os.path.basename(void_path)}, with shape {subject.shape})")
        input = torch.from_numpy(subject).squeeze(0).float()
        output = model(input)
        print(f"Output shape: {output.shape}")
        output_nib = nib.Nifti1Image(output.numpy(), affine=v.affine, header=v.header.copy())
        output_filename = os.path.join(output_dir, os.path.basename(void_path).replace("t1n-voided", ""))
        nib.save(output_nib, output_filename)
        print(f"Saved inpainted image to {output_filename}, shape: {output_nib.get_fdata().shape}")




'''
paired_transforms = Compose([
    LoadImaged(keys=["mri"]),
    EnsureChannelFirstd(keys=["mri"]),
    Spacingd(
        keys=["mri"],
        pixdim=(2, 2, 2),
        mode=("bilinear", "bilinear", "nearest")
    ),
    CenterSpatialCropd(keys=["mri"], roi_size=(128, 128, 80)),
    DivisiblePadD(keys=["mri"], k=16, mode="constant", constant_values=0),
    ScaleIntensityRangePercentilesd(
        keys=["mri"],
        lower=0,
        upper=99.5,
        b_min=-1.0,
        b_max=1.0
    ),
    ConcatItemsd(keys=["mri", "mask"], name="input", dim=0),
    EnsureTyped(keys=["input"]),
])

def get_dataloader(input, batch_size=1):
    dataset = Dataset(data=[], transform=paired_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

def load_case(void_path, mask_path):

    void = nib.load(void_path)
    mask = nib.load(mask_path)
    void_arr = void.get_fdata()
    mask_arr = mask.get_fdata()
    pair = np.stack([void_arr, mask_arr], axis=0) 

    return pair, void.affine, void.header.copy(), void

def run_inference(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    void_paths, mask_paths = read_paths_pair(input_dir)
    
    for void_path, mask_path in zip(void_paths, mask_paths):
        pair, affine, header, void = load_case(void_path, mask_path)
        pair = transforms(pair)

        input_tensor = torch.tensor(pair, dtype=torch.float32).unsqueeze(0).to(DEVICE)  
        model.eval()
        
        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor = output_tensor.cpu().squeeze(0).numpy()

        output_img = nib.Nifti1Image(output_tensor, affine=affine, header=header)
        output_filename = os.path.join(output_dir, os.path.basename(void_path).replace("t1n-voided", ""))
        output_resampled = resample_from_to(output_img, void, order=1)
        nib.save(output_resampled, output_filename)
        print(f"Saved inpainted image to {output_filename}, shape: {output_resampled.shape}")

if __name__ == "__main__":
    args   = argument_parser()
    net    = MedNextGenerator3D().to(DEVICE)
    net.load_state_dict(torch.load(args.model_weights, map_location=DEVICE))
    net.eval()
    run_inference(net, args.input_dir, args.output_dir)


    '''