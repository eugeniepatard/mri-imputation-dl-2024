# Deep Learning for MRI Image Inpainting

This project was developed as part of the **Deep Learning module** in the MSc *Applied Computational Science and Engineering* at **Imperial College London**. It was completed under a **24-hour assessment constraint**, simulating real-world time-limited project execution.

## 📘 Overview

In the medical field, Magnetic Resonance Imaging (MRI) scans often have missing data due to factors such as sensor issues or scan time limitations. This project explores deep learning-based methods for **image inpainting**—recovering missing portions of MRI images. Using PyTorch, a model is trained to restore missing parts of human head MRI images using artificially generated data for training and real corrupted test images.

## 🧠 Problem Statement

The project focuses on human head MRI scans, where a portion of the images is missing. We have access to a dataset of 100 human head images, but these images are corrupted by missing information. Our task is to design a neural network that can recover these missing portions of the images.

The dataset provided contains 100 corrupted MRI images, each of size 64x64 pixels. Additionally, we are given access to a generative model that can produce realistic-looking MRI images of human heads. Using this model, we will generate an appropriate training dataset.

The goal is to create a deep learning model that learns to recover missing portions of images and apply it to the corrupted test set.

## 🗂️ Project Structure

- `MRI-imputation-pipeline.ipynb`: Contains all analysis, code for generating the dataset, model design, training loop, and final predictions.
- `test_set_nogaps.npy`: Output file with the reconstructed MRI images for the test set.
- `References.md`: A record of all external tools and sources used during the project.
- `generated_images/`: Folder containing the artificially generated images for training.
- `test_set.npy`: The corrupted test dataset to be reconstructed.

## 📊 Data Features

- **Image Data**: MRI images of human heads of size 64x64 pixels.
- **Corrupted Data**: The test set contains images with missing regions.
- **Generated Data**: The training set contains images generated using a pre-trained generative model to fill in missing data.

## Methodology

The goal of this project is to implement and evaluate a generative model for generating and reconstructing images, leveraging a diffusion-based generative model, and utilizing image corruption techniques to simulate noisy or incomplete data. Below is a step-by-step breakdown of the methodology followed to achieve the desired results.

### 1. **Dataset Preparation**
To begin the project, the required datasets and files were downloaded and organized. The dataset file (`cw1_files.zip`) was obtained and uploaded to Google Drive for easy access. Once the file was available, the following steps were carried out:

- The `.zip` file containing the dataset and resources was extracted into the working directory.
- A setup script (`run.sh`) was executed to initialize the environment and install necessary dependencies for the project.
- A custom library (`ese_invldm`) was imported to facilitate the generation of samples using a pre-trained diffusion-based generative model.

### 2. **Diffusion-Based Generative Model for Image Generation**
The core of the project involves using a generative model based on diffusion to generate new images. The `generate` function from the `ese_invldm` library is employed to produce samples. Key parameters for generating images include:

- **num_samples**: Number of images to generate (2048 samples).
- **num_inference_steps**: The number of steps used during the inference phase (higher values improve image quality but require more computation).
- **batch_size**: Defines the number of images to generate per batch (64 images per batch).
- **scheduler**: Specifies the method for adjusting the sampling process, such as DDIM or DDPM.
- **temperature**: Controls randomness in the generated images (used to vary the diversity of generated samples).
- **seed**: A fixed random seed to ensure reproducibility of the image generation process.

The model was executed with 50 inference steps and a batch size of 64 to generate 2048 high-quality images. The images were then saved for further analysis.

### 3. **Corruption Pattern Generation**
The next step involved simulating image corruption using a checkered pattern. This corruption process is designed to model noisy data, mimicking the effect of missing or altered pixels in the images.

- A custom function, `generate_checked_pattern`, was defined to create a checkered pattern with alternating `0`s and `1`s. This pattern was applied to the images to simulate corruption.
- The pattern's block size and alignment were controlled by parameters such as `rect_width`, `rect_height`, `gap_width`, and `gap_height`. Adjustments to these parameters ensure a proper fit over the images.

The corruption pattern was applied to the generated images by element-wise multiplication with the original images, resulting in a set of corrupted images.

### 4. **Creating a Custom Dataset for Corrupted Images**
To facilitate model training and evaluation, a custom dataset class (`CustomDataset`) was created to load both the original and corrupted images.

- The dataset class loaded the generated images from the local directory and applied the corruption pattern to each image.
- During loading, transformations such as converting images into tensors were applied, making them compatible with machine learning models.
- Each sample returned by the dataset included both the corrupted version of the image and its original counterpart (used as the label). This allows models to learn how to reconstruct the original image from the corrupted input.

### 5. **Model Selection and Training**
Initially, a **Convolutional VAE (Variational Autoencoder)** was used to reconstruct the corrupted images. The VAE model, which is well-known for its ability to learn efficient latent representations of data, was trained to denoise the corrupted images. However, the performance of the Convolutional VAE was not satisfactory for this task, as it struggled to effectively reconstruct the original images from the corrupted versions.

After encountering limitations with the VAE, a **UNet** architecture was chosen as a more suitable alternative. The UNet architecture is renowned for its performance in image-to-image tasks, such as segmentation, denoising, and reconstruction. Its encoder-decoder structure with skip connections allows the model to preserve spatial information while progressively refining the image details.

- **UNet** was trained to reconstruct the original images from their corrupted versions. The training focused on minimizing the reconstruction error, enabling the model to learn how to restore the corrupted pixels and produce high-quality reconstructions.

The UNet model ultimately provided superior results in image restoration, outperforming the Convolutional VAE and demonstrating robustness in reconstructing the images from checkered corruptions.

---

This methodology outlines how a diffusion-based model was used to generate images, which were then corrupted and used to train a model for image reconstruction. After experimenting with a **Convolutional VAE** that did not perform well, a **UNet** architecture was selected for the reconstruction task, showing improved performance in image restoration. The project highlights the combination of generative models and corruption techniques to simulate real-world noisy data and train models for image denoising and reconstruction.


# Result

I experimented with several models before settling on the **U-Net architecture**, which proved to be the most effective for reconstructing images from corrupted inputs. Initially, I tried a **Convolutional VAE**, but it struggled with image reconstruction, and its loss curve was difficult to interpret. The **U-Net**, in contrast, provided much better results with smoother loss progression and more accurate reconstructions.

I also explored a **Wasserstein GAN (WGAN)**, but it was unstable, with a too-strong discriminator that resulted in poor image generation. As a result, I focused on the **U-Net**, which was easier to train and provided more reliable results.

After performing a **random search** for hyperparameters, I found the following configuration to yield the best performance:

- **Learning Rate** (`lr`): 0.0001
- **Weight Decay** (`wd`): 0.01
- **Betas**: (0.85, 0.98)

With these settings, the model performed well, and I saved the reconstructed images in the file `test_set_nogaps.npy`.

In conclusion, the **U-Net** architecture proved to be the best choice for this task, delivering high-quality image reconstruction, as demonstrated by the test set results.

## 🧑‍💻 Tools & Libraries

- Python, PyTorch
- Image Inpainting Techniques
- Generative Models for Data Augmentation
- Data Visualization (Matplotlib)
- NumPy, Pandas

## 🏫 Academic Context

This project was submitted as part of a **24-hour individual coursework** for the **Deep Learning** module in the **MSc in Applied Computational Science and Engineering** at **Imperial College London**. It reflects a practical, time-constrained application of deep learning techniques to solve real-world medical imaging challenges.
