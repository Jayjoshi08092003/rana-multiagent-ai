# 🎭 Real-Time Emotion Detection using RANA (Recursive Adaptive Neural Agent)

![GitHub repo size](https://img.shields.io/github/repo-size/Jayjoshi08092003/REAL_TIME_EMOTION_DETECTION)
![GitHub stars](https://img.shields.io/github/stars/Jayjoshi08092003/REAL_TIME_EMOTION_DETECTION?style=social)
![License](https://img.shields.io/github/license/Jayjoshi08092003/REAL_TIME_EMOTION_DETECTION)

This project showcases a real-time **emotion recognition system** developed as part of the **IIT Madras Research Park (IITMRP)** initiative. The system leverages **CNN architectures (ResNeXt-50, MiniXception)** and face detection with **YOLOv5**, all integrated within a modular AI agent called **RANA (Recursive Adaptive Neural Agent)**.

---

## 🧠 About RANA

**RANA** is designed as an adaptive emotional intelligence agent. It:

- Detects and classifies emotions from webcam video feed in real-time.
- Uses CNNs trained on the **FER-2013** dataset.
- Integrates **YOLOv5** for high-speed and accurate face detection.
- Supports real-time inference and dynamic routing for future multimodal systems.

---

## 🚀 Features

- Real-time facial emotion detection via webcam.
- Model training using **MiniXception** and **ResNeXt-50**.
- Supports CPU & GPU (CUDA).
- Webcam fallback compatibility (DSHOW, MSMF, V4L2).
- Live annotations using OpenCV.

---

## 🧪 Emotions Recognized

- Angry 😠  
- Disgust 🤢  
- Fear 😨  
- Happy 😄  
- Neutral 😐  
- Sad 😢  
- Surprise 😲

---

## 📁 Folder Structure

```

REAL\_TIME\_EMOTION\_DETECTION/
├── train\_model.py               # Train model on FER-2013 dataset
├── predict\_realtime.py         # Real-time webcam prediction script
├── model.py                    # MiniXception / ResNeXt model builder
├── utils.py                    # Accuracy, evaluation helpers
├── best\_model\_mini\_x.pth       # Trained model weights
├── FER-2013/                   # Dataset folder (train/test/val)
├── requirements.txt            # Required Python packages
└── README.md                   # This file

````

---

## 🧑‍💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Jayjoshi08092003/REAL_TIME_EMOTION_DETECTION.git
cd REAL_TIME_EMOTION_DETECTION
````

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (optional - model file already included)

```bash
python train_model.py
```

### 5. Run Real-Time Emotion Detection

```bash
python predict_realtime.py
```

> ⚠️ Make sure your webcam is connected and permissions are allowed.

---

## 🔍 Dataset

* **FER-2013**: Facial Expression Recognition 2013
* Public dataset with 7 labeled emotions
* Training: \~29k images | Size: 48x48 grayscale
* You can download it from [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 📦 Requirements

* Python 3.8+
* PyTorch
* torchvision
* OpenCV
* PIL
* NumPy

Install with:

```bash
pip install -r requirements.txt
```

---

## 🏗️ Model Architecture

* **MiniXception** for real-time light inference.
* Optional **ResNeXt-50** for high-accuracy offline training.
* Custom CNN wrappers for transfer learning.

---

## 📽️ Sample Output

<img src="assets/sample_output.jpg" width="600" alt="Sample Output Frame" />

---

## 📚 Citation

If this repo helps your research or work, please consider citing or giving a ⭐.

---

## 🤝 Acknowledgments

* [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [MiniXception](https://arxiv.org/abs/1710.07557)

---

## 📬 Contact

**Jay Joshi**
📧 [jayjoshi08092003@gmail.com](mailto:jayjoshi08092003@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/jayjoshi08092003/)
🔗 [GitHub](https://github.com/Jayjoshi08092003)


