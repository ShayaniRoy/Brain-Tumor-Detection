# Integrated Brain Tumor Detection & Medical Image Segmentation Pipeline

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, multi-stage computer vision and deep learning framework engineered to automate brain tumor region-of-interest (ROI) extraction, advanced preprocessing, and high-precision MRI scan classification.

---

# Integrated Brain Tumor Detection & Medical Image Segmentation Pipeline

An end-to-end, multi-stage computer vision and deep learning framework built to automate brain tumor ROI extraction, noise reduction, and MRI classification.

---

## 📌 Project Overview

This repository features an integrated automated diagnostic pipeline that merges classical computer vision preprocessing with deep learning for brain MRI scan analysis. The system isolates regions of interest (ROI) by removing artifacts and background noise before passing feature maps into a Convolutional Neural Network (CNN) for categorical classification.

---

## 🛠️ Tech Stack & Dependencies

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow, Keras |
| **Computer Vision** | OpenCV (`cv2`), Scikit-Image, PIL (Pillow), Rembg |
| **Data & Visualization** | NumPy, Matplotlib |

---


This repository presents a unified automated diagnostic pipeline designed to bridge classical computer vision techniques with deep learning methodologies for medical imaging analysis. The system systematically isolates diagnostically relevant areas by suppressing background noise, eliminating artifacts, and executing precise edge detection prior to inference via a Convolutional Neural Network (CNN).

---

## System Architecture & Processing Workflow

┌────────────────────────────────────────────────────────┐
│                   Raw Brain MRI Scan                   │
└───────────────────────────┬────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│     OpenCV Noise Reduction & Intensity Normalization   │
└───────────────────────────┬────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│          GrabCut & Rembg Artifact Extraction           │
└───────────────────────────┬────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│           Edge Detection & Feature Isolation           │
└───────────────────────────┬────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│        Deep CNN Classifier (TensorFlow / Keras)        │
└───────────────────────────┬────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│        Categorical Diagnostic Classification           │
└────────────────────────────────────────────────────────┘


---

## ✨ Key Technical Features

- **Automated Preprocessing & Segmentation**: Implements GrabCut segmentation, edge detection, and background removal via `rembg` for precise ROI extraction.
- **Deep Learning Classification**: Utilizes a Convolutional Neural Network (CNN) built in TensorFlow/Keras for multi-class classification.
- **End-to-End Execution**: Streamlines raw image ingestion, preprocessing, and inference into a cohesive Python execution model.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed on your system.

Install dependencies:

Bash
pip install tensorflow opencv-python rembg scikit-image numpy matplotlib
Run the pipeline:

Bash
python main.py

📄 License
Distributed under the MIT License. See LICENSE for details.














