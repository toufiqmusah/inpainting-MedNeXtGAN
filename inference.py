"""inference.py - Using NiBabel for resampling"""

import os
import torch
import argparse
from glob import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from config import DEVICE
from models import MedNextGenerator3D
from monai.transforms import (
    Compose,
    Spacingd,
    LoadImaged,
    EnsureTyped,
    ConcatItemsd,
    Orientationd,
    DivisiblePadD,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd)
from monai.data import Dataset, DataLoader

def argument_parser():
    parser = argparse.ArgumentParser(description="Inference script for MRI style transfer")
    parser.add_argument("--model_weights", type=str, required=True, help="Model Weights Path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing voided MRI files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for inpainted results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    return parser.parse_args()

def get_base_filename(void_path):
    basename = os.path.basename(void_path)
    base_name = basename.replace('t1n-voided.nii.gz', '.nii.gz')
    return base_name

def read_paths_pair(root_dir: str):
    void_mri_files = sorted(glob(os.path.join(root_dir, "**", "*t1n-voided.nii.gz"), recursive=True))
    mask_files = sorted(glob(os.path.join(root_dir, "**", "*mask.nii.gz"), recursive=True))
    
    data_pairs = []
    for void, mask in zip(void_mri_files, mask_files):
        base_name = get_base_filename(void)
        data_pairs.append({
            "mri": void, 
            "mask": mask,
            "output_name": base_name,
            "original_path": void
        })
    
    return data_pairs

def resample_to_original_space(pred_array, original_nii_path, processed_shape=(128, 128, 80)):
    """
    Resample prediction back to original image space using NiBabel and scipy
    
    Args:
        pred_array: numpy array of prediction (H, W, D)
        original_nii_path: path to original NIfTI file
        processed_shape: shape used during processing
    
    Returns:
        resampled_nii: NiBabel NIfTI image in original space
    """
    # Load original image
    original_nii = nib.load(original_nii_path)
    original_data = original_nii.get_fdata()
    original_shape = original_data.shape
    
    print(f"Resampling from {pred_array.shape} to {original_shape}")
    
    # Calculate zoom factors for each dimension
    zoom_factors = [
        original_shape[i] / pred_array.shape[i] 
        for i in range(3)
    ]
    
    print(f"Zoom factors: {zoom_factors}")
    
    # Resample using scipy zoom (trilinear interpolation)
    resampled_data = zoom(pred_array, zoom_factors, order=1, mode='nearest')
    
    # Ensure the resampled data has exactly the original shape
    if resampled_data.shape != original_shape:
        print(f"Adjusting shape from {resampled_data.shape} to {original_shape}")
        # If there are small differences, crop or pad
        final_data = np.zeros(original_shape, dtype=resampled_data.dtype)
        
        # Calculate slicing for each dimension
        slices = []
        for i in range(3):
            if resampled_data.shape[i] <= original_shape[i]:
                # If smaller, center it
                start = (original_shape[i] - resampled_data.shape[i]) // 2
                end = start + resampled_data.shape[i]
                slices.append((slice(start, end), slice(None)))
            else:
                # If larger, crop from center
                start = (resampled_data.shape[i] - original_shape[i]) // 2
                end = start + original_shape[i]
                slices.append((slice(None), slice(start, end)))
        
        # Apply slicing
        final_data[slices[0][0], slices[1][0], slices[2][0]] = \
            resampled_data[slices[0][1], slices[1][1], slices[2][1]]
        
        resampled_data = final_data
    
    # Create new NIfTI image with original header and affine
    resampled_nii = nib.Nifti1Image(
        resampled_data.astype(original_data.dtype), 
        original_nii.affine, 
        original_nii.header
    )
    
    return resampled_nii

# Simplified transforms for inference
paired_transforms = Compose([
    LoadImaged(keys=["mri", "mask"]),
    EnsureChannelFirstd(keys=["mri", "mask"]),
    Orientationd(keys=["mri", "mask"], axcodes="RAS"),
    Spacingd(
        keys=["mri", "mask"],
        pixdim=(2, 2, 2),
        mode=("bilinear", "nearest")
    ),
    CenterSpatialCropd(keys=["mri"], roi_size=(128, 128, 80)),
    DivisiblePadD(keys=["mri", "mask"], k=16, mode="constant", constant_values=0),
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

def get_dataloader(input_dir, batch_size=1):
    data_files = read_paths_pair(input_dir)
    paired_dataset = Dataset(data=data_files, transform=paired_transforms)
    paired_loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=False)
    return paired_loader, data_files

def inference(model, input_dir, output_dir, batch_size=1):
    output_dir = os.path.join(output_dir, "inference_results")
    os.makedirs(output_dir, exist_ok=True)
    
    data_loader, data_files = get_dataloader(input_dir, batch_size=batch_size)
    model.eval()
    
    print(f"Starting inference on {len(data_files)} files...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_img = batch["input"].to(DEVICE)
            pred = model(input_img)
            
            for i in range(pred.shape[0]):
                pred_single = pred[i, 0].cpu().numpy()  # Remove batch and channel dims
                
                file_idx = batch_idx * batch_size + i
                if file_idx < len(data_files):
                    output_name = data_files[file_idx]["output_name"]
                    original_path = data_files[file_idx]["original_path"]
                    output_path = os.path.join(output_dir, output_name)
                    
                    try:
                        # Resample prediction back to original space
                        resampled_nii = resample_to_original_space(
                            pred_single, 
                            original_path, 
                            processed_shape=(128, 128, 80)
                        )
                        
                        # Save resampled result
                        nib.save(resampled_nii, output_path)
                        print(f"Successfully saved: {output_name}")
                        
                    except Exception as e:
                        print(f"Error processing {output_name}: {e}")
                        
                        # Fallback: save without resampling
                        try:
                            basic_nii = nib.Nifti1Image(pred_single, np.eye(4))
                            fallback_path = os.path.join(output_dir, f"fallback_{output_name}")
                            nib.save(basic_nii, fallback_path)
                            print(f"Fallback save: fallback_{output_name}")
                        except Exception as e2:
                            print(f"Fallback also failed for {output_name}: {e2}")

def main():
    args = argument_parser()
    
    model = MedNextGenerator3D(input_channels=2, output_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(args.model_weights, map_location=DEVICE))
    
    print(f"Model loaded from: {args.model_weights}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Run inference
    inference(model, args.input_dir, args.output_dir, args.batch_size)
    print("Inference completed!")

if __name__ == "__main__":
    main()