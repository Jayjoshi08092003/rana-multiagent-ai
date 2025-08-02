
# 🤖 RANA: Recursive Adaptive Neural Agent

**RANA** (Recursive Adaptive Neural Agent) is an intelligent multimodal AI system developed to classify and interpret human facial emotions in real time. RANA fuses powerful deep learning architectures with adaptive learning techniques for robust emotion recognition.

---

## 🚀 Overview

RANA combines:
- Convolutional Neural Networks (CNNs) like **ResNeXt-50** and **MiniXception**
- A facial emotion dataset (**FER-2013**)
- Real-time webcam integration
- YOLOv5 for efficient face detection
- Custom adaptive learning logic

This allows RANA to perform:
- 🎯 Emotion classification with 7 key emotions
- 🖼️ Face detection and cropping
- 🔁 Self-improvement via adaptive learning loops
- 📦 Future scalability to multimodal contexts (e.g., text + vision)

---

## 🔍 Emotion Classes

- Angry 😠  
- Disgust 🤢  
- Fear 😨  
- Happy 😄  
- Neutral 😐  
- Sad 😢  
- Surprise 😲  

---

## 🛠️ Core Components

### 📦 CNN Backbone  
- Initially trained using **ResNeXt-50**
- Deployed in real-time with **MiniXception** for speed and efficiency

### 🎯 Training  
- Dataset: FER-2013 (grayscale, 48x48 images)
- Data Augmentation: Horizontal flips, normalization
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  

### 📷 Real-Time Detection  
- Face detection via **YOLOv5**
- Emotion prediction from webcam streams
- Deployment-ready via OpenCV

### 🧠 Adaptive Learning Loop  
- Dynamically adjusts learning rate based on validation loss
- Tracks best model using checkpointing
- Can be extended for online learning or reinforcement learning

---

## 📁 Repository Structure

\`\`\`
RANA/
├── model.py                  # Model definition (MiniXception/ResNeXt)
├── train_model.py            # Emotion classifier training script
├── realtime_emotion_detect.py # Real-time webcam inference
├── datasets/                 # FER-2013 data (or path to it)
├── best_model.pth            # Trained model checkpoint
├── README.md                 # Project overview (this file)
\`\`\`

---

## 💡 Applications

- Human-computer interaction  
- Affective computing  
- Virtual assistants and robots  
- Education technology (edtech) feedback systems  
- Health monitoring tools

---

## 🧩 Future Roadmap

- 🧠 Integrate text + audio for full multimodal emotional AI  
- 🌍 Add multilingual support  
- ⚙️ Export as REST API or ONNX for mobile deployment  
- 📚 Fine-tune with real-world emotion datasets

---

## 📬 Contact

**👨‍💻 Developer:** Jay Joshi  
📧 Email: jayjoshi08092003@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jayjoshi08092003)  
🔗 [GitHub](https://github.com/Jayjoshi08092003)

---

## 📄 License

Licensed under the [MIT License](LICENSE).

---

## 🏷️ Tags

`#RANA` `#AI`  `#EmotionRecognition` `#DeepLearning` `#RealTimeAI` `#FER2013`


