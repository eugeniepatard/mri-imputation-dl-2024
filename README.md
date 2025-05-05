# Deep Learning for MRI Image Inpainting

This project was developed as part of the **Deep Learning module** in the MSc *Applied Computational Science and Engineering* at **Imperial College London**. It was completed under a **24-hour assessment constraint**, simulating real-world time-limited project execution.

## 📘 Overview

In the medical field, Magnetic Resonance Imaging (MRI) scans often have missing data due to factors such as sensor issues or scan time limitations. This project explores deep learning-based methods for **image inpainting**—recovering missing portions of MRI images. Using PyTorch, a model is trained to restore missing parts of human head MRI images using artificially generated data for training and real corrupted test images.

## 🧠 Problem Statement

The project focuses on human head MRI scans, where a portion of the images is missing. We have access to a dataset of 100 human head images, but these images are corrupted by missing information. Our task is to design a neural network that can recover these missing portions of the images.

The dataset provided contains 100 corrupted MRI images, each of size 64x64 pixels. Additionally, we are given access to a generative model that can produce realistic-looking MRI images of human heads. Using this model, we will generate an appropriate training dataset.

The goal is to create a deep learning model that learns to recover missing portions of images and apply it to the corrupted test set.

## 🗂️ Project Structure

- `Assessment.ipynb`: Contains all analysis, code for generating the dataset, model design, training loop, and final predictions.
- `test_set_nogaps.npy`: Output file with the reconstructed MRI images for the test set.
- `References.md`: A record of all external tools and sources used during the project.
- `generated_images/`: Folder containing the artificially generated images for training.
- `test_set.npy`: The corrupted test dataset to be reconstructed.

## 📊 Data Features

- **Image Data**: MRI images of human heads of size 64x64 pixels.
- **Corrupted Data**: The test set contains images with missing regions.
- **Generated Data**: The training set contains images generated using a pre-trained generative model to fill in missing data.

## ⚙️ Methodology

1. **Data Generation and Preprocessing**  
   - Utilized a generative model to create realistic MRI images of human heads.  
   - Saved these generated images for use in training the model.  
   - Loaded and displayed 10 generated images and 10 corrupted images from the `test_set.npy` for visualization.

2. **Dataset Construction**  
   - Created a PyTorch `TensorDataset` for both the training and test datasets.  
   - For the training set, used generated images and artificially corrupted them to simulate real-world missing data.  
   - The test set contains corrupted images with no corresponding labels.  
   - Used PyTorch `DataLoader` to load the datasets in batches for training.

3. **Model Design**  
   - Designed a neural network architecture based on convolutional layers to recover missing portions of images.  
   - The model uses an encoder-decoder architecture where the encoder extracts features from the corrupted image and the decoder reconstructs the missing parts.  
   - Defined a custom loss function (Mean Squared Error) to compare the model's predictions with the ground truth images.

4. **Training**  
   - Trained the model on the training set for 50 epochs using the Adam optimizer with a learning rate of 1e-4.  
   - Monitored the loss during training and validated the model using a separate validation set.

5. **Evaluation & Output**  
   - Evaluated the model on the test set, displaying 10 images with recovered missing portions.  
   - Saved the test data with the missing portions filled in and stored it as `test_set_nogaps.npy`, ensuring the order matches the original `test_set.npy`.

## 📊 Results

- The model was able to reconstruct missing portions of the MRI images with reasonable accuracy.  
- Visual comparisons between the corrupted and recovered images demonstrated significant improvements in missing data regions.  
- While the model performed well, some edge cases with large missing portions were less accurate, highlighting areas for further improvement.

- The final model produced reconstructions that closely resembled the ground truth images, and the predictions were coherent with the context of the MRI scan.

## 🧑‍💻 Tools & Libraries

- Python, PyTorch
- Image Inpainting Techniques
- Generative Models for Data Augmentation
- Data Visualization (Matplotlib)
- NumPy, Pandas

## 🏫 Academic Context

This project was submitted as part of a **24-hour individual coursework** for the **Deep Learning** module in the **MSc in Applied Computational Science and Engineering** at **Imperial College London**. It reflects a practical, time-constrained application of deep learning techniques to solve real-world medical imaging challenges.
