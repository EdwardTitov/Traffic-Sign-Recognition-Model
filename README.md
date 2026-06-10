# 🚦 SignalNet: Lightweight Traffic Sign Detection & Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://github.com/tensorflow/tensorflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Computer Vision pipeline featuring a custom, ultra-lightweight Convolutional Neural Network (CNN) optimized for real-time traffic sign classification. Built specifically to balance high accuracy with rigid edge-computing constraints.

---

## 📌 Project Overview

Deploying deep learning models to autonomous vehicle edge devices requires a strict balance between memory footprint, computational latency, and predictive accuracy. This project delivers a custom CNN architecture designed from scratch to solve the traffic sign classification problem without the heavy overhead of standard transfer learning backbones (like ResNet or MobileNet). 

By leveraging efficient feature extraction layers, strategic downsampling, and aggressive regularization, this model achieves near-state-of-the-art performance while remaining compact enough to run locally on low-power embedded systems.

---

## 📊 Performance & Key Metrics

The core design philosophy of this architecture was **maximal efficiency**. Below is a summary of the production performance benchmarks:

| Metric | Value | Target Benchmark / Context |
| :--- | :--- | :--- |
| **Model Size** | **134k parameters** (~536 KB unquantized) | Ideal for low-memory edge devices |
| **Accuracy** | **93.0%** | Evaluated on unseen, highly diverse test data |
| **Median Latency** | **62 ms** | Tested on standard CPU baseline (Batch Size = 1) |
| **Data Streaming** | **Deeplake Enabled** | Zero-local-storage required for dataset hosting |

### Core Features
* **Streaming Architecture:** Integrated with DeepLake for cloud-native data streaming, eliminating the need to download massive datasets locally during training iterations.
* **Production Logging & Analytics:** Includes full evaluation suites powered by `scikit-learn`, `Matplotlib`, and `Seaborn` to generate confusion matrices and ROC curves automatically.
* **Edge-Ready Backbone:** Highly modular TensorFlow/Keras implementation making it trivial to export to TensorFlow Lite (TFLite) or ONNX formats.

---

## 🛠️ Tech Stack

* **Core ML Framework:** `Python`, `TensorFlow / Keras`
* **Data Management:** `Deeplake` (Activeloop)
* **Mathematical Operations:** `NumPy`, `scikit-learn`
* **Visualization & Diagnostics:** `Matplotlib`, `Seaborn`

---

## 🚀 Installation & Usage

### 1. Prerequisites & Environment Setup
Clone the repository and set up a isolated virtual environment to prevent dependency conflicts:

```bash
# Clone the repository
git clone [https://github.com/yourusername/traffic-sign-detection.git](https://github.com/yourusername/traffic-sign-detection.git)
cd traffic-sign-detection

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
