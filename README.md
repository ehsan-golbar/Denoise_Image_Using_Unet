# Denoise_Image_Using_Unet

This project implements a Convolutional Neural Network (CNN) using the U-Net architecture to remove salt and pepper noise from images. The model is designed to learn and reconstruct noise-free images while preserving structural details.  

---

## Dataset and Preprocessing  
- **Dataset:** Consists of 36,000 records.  
- Extracted images from the 'pixels' column and prepared them for the model.  
- Split into three sets:  
  - Training  
  - Validation (PrivateTest)  
  - Test (PublicTest)  
- Applied **salt and pepper noise** to images.  
- Normalized pixel values to the range [0,1].  
- Converted images to formats compatible with **PyTorch** and loaded using `DataLoader`.  

---

## Model Architecture  
The U-Net model is designed with two main parts:  
1. **Encoder:** Extracts high-level features from the noisy input image.  
2. **Decoder:** Reconstructs the original image from the extracted features.  

- **ConvModule Class:**  
  - Contains two Conv2D layers for feature extraction.  
  - Uses `LeakyReLU` for non-linearity.  
  - `BatchNorm2D` for output normalization and learning stability.  
  - `Dropout` layer to reduce overfitting.  

- **UpConvModule Class:**  
  - Upsamples features using `ConvTranspose2D`.  
  - Utilizes **Skip Connections** to combine encoder features with decoder layers for better reconstruction.  

- **DenoisingUNet Class:**  
  - Implements the complete U-Net architecture.  
  - Composed of:  
    - 5 Encoder blocks  
    - 4 Decoder blocks  
    - 1 Output layer  

---

## Training and Evaluation  
- **Loss Function:** Mean Squared Error (MSE Loss) to measure pixel-wise differences.  
- **Optimizer:** Adam with a learning rate of 0.001.  
- **Training Process:**  
  - Trained for 5 epochs on training and validation data.  
  - Saved the best model based on the lowest validation loss.  

### Performance Metrics:  
- **PSNR (Peak Signal-to-Noise Ratio):**  
  - Achieved **35 dB**, indicating high reconstruction quality.  
- **SSIM (Structural Similarity Index):**  
  - Value: **0.9726**, reflecting high structural similarity with the original image.  
- **MSE (Mean Squared Error):**  
  - Value: **0.000286**, showing minimal pixel difference.  

---

## How to Run  
1. **Install Dependencies**  
   ```bash
   pip install torch torchvision numpy matplotlib scikit-image

