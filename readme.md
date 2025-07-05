# Inpainting-MedNeXtGAN

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/toufiqmusah/inpainting-MedNeXtGAN.git
```

### 2. Install Requirements
```bash
pip install -r inpainting-MedNeXtGAN/requirements.txt
```

### 3. Download Dataset

A dataset download script is provided:

```bash
cd inpainting-MedNeXtGAN
sed -i 's/\r$//' dataset-download.sh
chmod +x dataset-download.sh
./dataset-download.sh
cd ..
```

## Training

```bash
python inpainting-MedNeXtGAN/main.py \
  --input_dir "<TRAIN_DATA_DIR>" \
  --batch_size 2 \
  --num_epochs 500
```

**Parameters:**
- `<TRAIN_DATA_DIR>`: Directory with your training set (e.g., `inpainting-MedNeXtGAN/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training`)

## Inference

After training, run inference on validation data:

```bash
python inpainting-MedNeXtGAN/inference.py \
  --model_weights "<MODEL_WEIGHTS>" \
  --input_dir "<VAL_DATA_DIR>" \
  --output_dir "<OUTPUT_DIR>"
```

**Parameters:**
- `<MODEL_WEIGHTS>`: Path to your trained weights file (e.g., `G31.pth`)
- `<VAL_DATA_DIR>`: Directory with your validation/test data
- `<OUTPUT_DIR>`: Output directory for generated images

## Directory Structure

```
inpainting-MedNeXtGAN/
│
├── main.py                 # Training script
├── inference.py            # Inference script
├── dataset-download.sh     # Dataset download script
├── requirements.txt        # Python dependencies
├── models.py              # Model architectures
├── utils.py               # Additional modules and scripts
└── data.py                # Data preprocessing, Loaders
```