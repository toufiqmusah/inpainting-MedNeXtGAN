gdown -q 1-lKBKHDEGL3oD2TFJ4-B7e8lmef3KxuZ # --output '../' # in-painting training
# !gdown -q 1--knKafAmI3F2N_db0L4CrWo7rXsL63s # in-painting validation

unzip -q ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training.zip
# !unzip -q ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation.zip

echo "All done preparing test sets!"

# !sed -i 's/\r$//' dataset-download.sh
# !chmod +x dataset-download.sh
# !./dataset-download.sh