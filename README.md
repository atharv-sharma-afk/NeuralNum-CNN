# 🖋️ Interactive MNIST Digit Classifier

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

An end-to-end deep learning application that recognizes handwritten digits in real-time. This project features a custom Convolutional Neural Network (CNN) trained on the MNIST dataset using **PyTorch**, paired with an interactive frontend built in **Streamlit**.

Unlike standard "black box" models, this application extracts and visualizes the intermediate feature maps (activations) to show exactly how the CNN interprets the image layer by layer.

## ✨ Features
* **Custom CNN Architecture:** A lightweight, highly accurate PyTorch model utilizing 3x3 convolutions, Max Pooling, and ReLU activations.
* **Real-Time Inference:** Draw a digit on the Streamlit canvas and get instant probability scores for all 10 classes.
* **Inside the Black Box:** Visualizes the network's internal feature maps (Conv1 and Conv2 layers) during inference to demonstrate spatial pattern recognition.
* **Robust Training:** Implements data augmentation (random rotation and affine transformations) to prevent overfitting and improve real-world accuracy.

## 🧠 Model Architecture

The network processes a 1x28x28 grayscale image through the following pipeline:

1.  **Conv Block 1:** `Conv2d(1 -> 16, 3x3, padding=1)` → `ReLU` → `MaxPool2d(2x2)` (Output: 16x14x14)
2.  **Conv Block 2:** `Conv2d(16 -> 32, 3x3, padding=1)` → `ReLU` → `MaxPool2d(2x2)` (Output: 32x7x7)
3.  **Flatten:** Converts the 2D feature maps into a 1D array of 1568 elements.
4.  **Fully Connected 1:** `Linear(1568 -> 128)` → `ReLU`
5.  **Output Layer:** `Linear(128 -> 10)` (Outputs raw logits for Cross-Entropy evaluation)

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
