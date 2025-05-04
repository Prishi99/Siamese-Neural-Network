# Siamese Neural Network on MNIST Dataset

## 🧠 Project Overview

This project implements a **Siamese Neural Network** to perform image similarity detection using the **MNIST dataset** of handwritten digits. The network is trained to determine whether two images represent the same digit, making it a binary classification task: similar (1) or dissimilar (0).

---

## 🔍 What is a Siamese Neural Network?

A **Siamese Neural Network (SNN)** is a type of neural network architecture that learns to differentiate between pairs of inputs by comparing their features. It consists of **two identical subnetworks** (usually CNNs) that share weights and parameters. Each subnetwork extracts features from one input, and the **distance between the outputs** (often using Euclidean distance) determines the similarity between the inputs.

### 👨‍🔬 How it works:
1. Input: A pair of images (Image A and Image B)
2. Each image is passed through the same convolutional neural network (CNN) to extract feature embeddings.
3. The feature vectors are compared using a distance function (e.g., Euclidean or contrastive loss).
4. The network is trained to minimize the distance for similar images and maximize it for dissimilar ones.

---

## 🧪 Dataset: MNIST

The **MNIST dataset** is a standard benchmark in computer vision, containing 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels.

In this project:
- Training pairs are generated as either **positive pairs** (same digit) or **negative pairs** (different digits).
- These pairs are used to train the Siamese network to understand digit similarity.

---

## ⚙️ Technologies Used

- Python
- TensorFlow / PyTorch
- NumPy
- Matplotlib
- Scikit-learn (for metrics and evaluation)

---



## 🎯 Applications of Siamese Neural Networks

Siamese Networks are widely used in tasks involving **similarity learning**, such as:

- 🔐 **Face verification** (e.g., Face ID): Verifying whether two images belong to the same person.
- 📄 **Signature verification**: Detecting forgery by comparing handwritten signatures.
- 🏥 **Medical image matching**: Identifying similar medical conditions or anomalies.
- 📷 **One-shot learning**: Classifying new data with very few examples.
- 🛍️ **Product recommendation**: Finding visually or semantically similar items.

---

## 📊 Results

The trained model achieves high accuracy in determining whether a pair of MNIST images are similar or not. Visualization of positive and negative pairs shows the model effectively learns meaningful features for digit similarity.

---


## 📚 References

- [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- TensorFlow and PyTorch documentation
