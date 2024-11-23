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

## Training
If you want to train your own model, you can follow the steps below:
1. Change the configuration in `config.py` to your liking. (Skip the model parameters because I used a model from package `segmentation_models_pytorch`).
2. Change the save repository in Hugging Face part in `main.py`.
3. Go to Kaggle or some other platform to train the model.
4. Clone the models on the Kaggle platform and install the necessary packages.
   ```bash
    pip install -r requirements.txt
    ```
5. Run the `main.py` script to train the model.
    ```bash
    python3 train.py --huggingface_token <your_huggingface_token> --wandb_key <your_wandb_key>
    ```
    The models will be saved in the Hugging Face repository you specified in the `main.py` script.