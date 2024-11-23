# BKAI-IGH-NeoPolyp-Unet-model

This project is for the development of a polyp segmentation model for the BKAI-IGH-NeoPolyp dataset. The model is based on the U-Net architecture and is trained using the PyTorch framework.

## Inference

1. Download the best model I have trained from Hugging Face by using terminal (Just run it in the root directory of this project):
    ```bash
    wget -O model.pth https://huggingface.co/auphong2707/BKAI-IGH-NeoPolyp-Unet-model/resolve/main/experiment_0/best.pth?download=true
    ```
2. Run the inference script:
    ```bash
    python3 infer.py --image_path image.jpeg
    ```
    For simplicity, the image will be saved in the root directory of this project with the same name as the input image but with the suffix `_pred`.