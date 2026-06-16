# 🚦 SignalNet: Lightweight Traffic Sign Classification with TensorFlow

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://github.com/tensorflow/tensorflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Convolutional Neural Network (CNN) for real-time traffic sign classification built with TensorFlow/Keras and trained on the German Traffic Sign Recognition Benchmark (GTSRB). The model prioritizes fast inference and a small memory footprint while maintaining strong classification accuracy.

---

## 📌 Project Overview

Traffic sign recognition is a core component of modern Advanced Driver Assistance Systems (ADAS) and autonomous driving pipelines. While large transfer-learning models can achieve high accuracy, they often introduce significant computational and memory overhead.

SignalNet demonstrates that a compact CNN can achieve strong performance with:

- Low parameter count
- Sub-millisecond CPU inference latency
- Minimal preprocessing requirements
- Straightforward deployment to edge environments

The project uses DeepLake to stream the GTSRB dataset directly from the cloud without requiring local dataset storage.

---

## 📊 Results

| Metric | Value |
|----------|----------|
| Test Accuracy | **93.76%** |
| Test Loss | **0.2727** |
| Median Inference Latency | **0.75 ms** |
| Average Inference Latency | **0.92 ms** |
| Minimum Latency | **0.60 ms** |
| Maximum Latency | **3.10 ms** |
| Total Parameters | **44,715** |
| Input Resolution | **26×26×3** |
| Number of Classes | **43** |

### Training Performance

```text
Epoch 12/12
Accuracy: 99.99%
Training Loss: 0.0028
```

### Test Performance

```text
Accuracy: 93.76%
Loss: 0.2727
```

---

## 🗂 Dataset

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) hosted through DeepLake.

### Dataset Details

- 43 traffic sign classes
- 39,209 training images
- 12,630 test images
- RGB images with varying resolutions
- Images resized to 26×26 before training

Datasets are streamed directly from DeepLake:

```python
train_ds = dl.load("hub://activeloop/gtsrb-train")
test_ds = dl.load("hub://activeloop/gtsrb-test")
```

---

## 🧠 Model Architecture

The network is intentionally compact:

```text
Input (26×26×3)
│
├── Conv2D (32 filters, 3×3, ReLU)
├── MaxPooling2D
│
├── Conv2D (32 filters, 3×3, ReLU)
├── BatchNormalization
├── MaxPooling2D
│
├── Flatten
│
└── Dense (43, Softmax)
```

### Parameter Breakdown

| Layer | Parameters |
|---------|---------:|
| Conv2D | 896 |
| Conv2D | 9,248 |
| BatchNormalization | 128 |
| Dense | 34,443 |
| **Total** | **44,715** |

---

## ⚙️ Training Configuration

### Optimizer

```python
keras.optimizers.Adam()
```

### Learning Rate Schedule

```python
keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=10000
)
```

### Loss Function

```python
sparse_categorical_crossentropy
```

### Hyperparameters

| Parameter | Value |
|----------|----------|
| Epochs | 12 |
| Batch Size | 128 |
| Initial Learning Rate | 0.001 |
| Scheduler | Cosine Decay |
| Optimizer | Adam |

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/signalnet.git
cd signalnet
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

Train the model:

```bash
python train.py
```

The training pipeline:

1. Streams GTSRB via DeepLake
2. Resizes images to 26×26
3. Normalizes pixel values to `[0,1]`
4. Trains for 12 epochs
5. Evaluates on the held-out test set

---

## 📈 Evaluation

Evaluate the model:

```bash
python evaluate.py
```

Outputs include:

- Test accuracy
- Test loss
- Latency statistics
- Confusion matrix visualization

---

## ⏱ Latency Benchmarking

Inference latency is measured after a warm-up phase using TensorFlow graph execution:

```python
compiled_model = tf.function(model)
```

### Benchmark Configuration

- Warmup samples: 20
- Measurement samples: 500
- Batch size: 1

### Results

```text
Average Latency: 0.92 ms
Median Latency: 0.75 ms
Minimum Latency: 0.60 ms
Maximum Latency: 3.10 ms
```

---

## 📊 Confusion Matrix

A normalized confusion matrix is generated using:

```python
from sklearn.metrics import confusion_matrix
```

Visualization is created using:

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
sns.heatmap(cm_norm, cmap="Blues")
```

Add your generated figure to:

```text
assets/confusion_matrix.png
```

and include it below:

![Confusion Matrix](assets/confusion_matrix.png)

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- DeepLake
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## 📁 Project Structure

```text
signalnet/
│
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── confusion_matrix.png
│
└── notebooks/
    └── signalnet.ipynb
```

---

## 🔮 Future Improvements

- [ ] Data augmentation pipeline
- [ ] TensorFlow Lite export
- [ ] INT8 quantization
- [ ] ONNX deployment
- [ ] Raspberry Pi benchmarking
- [ ] Real-time webcam traffic sign recognition
- [ ] Additional regularization techniques

---

## 📄 License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
