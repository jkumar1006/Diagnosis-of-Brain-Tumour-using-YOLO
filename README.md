# 🧠 Brain Tumor Detection using YOLO & CNN

A deep learning-based medical imaging project that uses **YOLOv4** for real-time object detection and **Convolutional Neural Networks (CNN)** for classification of brain tumors in MRI scans. This system is designed to assist healthcare professionals in early detection and accurate classification of brain tumors.

---

## 📌 Overview

This project combines the power of object detection and classification to build a robust model for identifying and categorizing brain tumors. The architecture includes image preprocessing, YOLO-based detection, and CNN-based classification into benign or malignant tumors.

---

## 🔍 Features

- 🎯 High-speed and accurate detection with YOLOv4.
- 🧠 Tumor classification into **benign** or **malignant**.
- 📷 Image preprocessing with thresholding, filtering, and masking.
- 📈 Performance evaluation through accuracy, precision, recall, and ROC curves.

---

## 🛠️ Technology Stack

| Category               | Tools/Technologies Used                    |
|------------------------|--------------------------------------------|
| Programming Language   | Python, MATLAB                             |
| Libraries/Frameworks   | TensorFlow, Keras, OpenCV, Darknet (YOLO)  |
| Visualization          | Matplotlib, Seaborn                        |
| Modeling Techniques    | CNN, YOLOv4, Residual Blocks               |

---

## 📂 Dataset

- Total Images: **10,000+** MRI scans
- Categories: **Benign** and **Malignant** tumors
- Format: **RGB**, **Grayscale**
- Sources: Open-access datasets from [Kaggle](https://www.kaggle.com/) and academic repositories

---

## 📊 Model Evaluation

| Model           | Accuracy (%) |
|----------------|--------------|
| YOLO + CNN     | 94–96        |
| Random Forest  | 86           |
| Decision Tree  | 75           |
| KNN            | 73           |
| ANN            | 72           |

---

## 🚀 How to Use
1. **Clone the Repository**
git clone https://github.com/yourusername/brain-tumor-detection-yolo-cnn.git
cd brain-tumor-detection-yolo-cnn

2. **Set up Environment**
pip install -r requirements.txt

3. **Train YOLOv4 Model**

   - Label your dataset with bounding boxes.

   - Modify the YOLO config files as per your dataset.

   - Use Darknet or a Python wrapper to begin training.

4. **Run Detection**
python detect.py --image ./test/image1.jpg

5. **Classify Tumor**
python classify.py --image ./detected/image1.jpg

6. **View Results**

   - Predictions and labels will be shown on the image.

   - Performance metrics will be saved as graphs and tables.

---
