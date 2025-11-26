# TA-2 Assignment: DeepFake Image Detection Model

Name: Utsav Nagar
Enrollment No: 250103002026
Course: MSc Cyber Security

This project implements a complete DeepFake Image Detection System using:

TensorFlow / Keras

MobileNetV3-Small (Fine-tuned)

Mediapipe Face Detection

Custom DeepFake dataset (merged from multiple folders)

TFLite conversion for deployment

HuggingFace Space compatibility

VS Code + Local Training Support


---

## 📁 Project Structure

```
FakeImageDetectionModelTraining/
│
├── real_and_fake_face/
│   ├── training_real/
│   └── training_fake/
│
├── real_and_fake_face_detection/
│   └── real_and_fake_face/
│       ├── training_real/
│       └── training_fake/
│
├── dataset/                # Auto-generated merged dataset (real/fake)
├── dataset_cropped/        # Auto-generated face-cropped dataset
│
├── models/
│   ├── deepfake_model.h5
│   └── deepfake_model.tflite
│
├── src/
│   └── train.py            # FULL training pipeline
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🚀 1. Dataset Processing Pipeline

Your raw dataset exists in two different folders:

```
real_and_fake_face/
real_and_fake_face_detection/real_and_fake_face/

```
Each contains:
```
training_real/
training_fake/

```
🔧 Step 1 — Auto-merge dataset

The training script automatically merges all real/fake images into:
```
dataset/
    real/
    fake/

```

🔧 Step 2 — Face Detection + Cropping
Using Mediapipe, each image is cropped to remove background noise:
```
dataset_cropped/
    real/
    fake/
```

---

# 🧠 2. Model Training Pipeline (train.py)
The src/train.py file performs:

✔ Dataset Merge

✔ Face Detection (Mediapipe)
✔ Data Augmentation
✔ MobileNetV3-Small Fine-tuning
✔ Accuracy/Loss Graphs
✔ Confusion Matrix
✔ Classification Report
✔ Saving .h5 and .tflite models

🔥 Training Model

Input Size: 224×224×3

Base model: MobileNetV3Small (ImageNet pretrained)

Trainable layers: Enabled (fine-tuning)

Optimizer: Adam (1e-4)

Epochs: 15

Loss: Binary Cross-Entropy
### 🏁 Output Models
```
models/deepfake_model.h5        # Full TF model
models/deepfake_model.tflite    # Mobile/Edge version
```
### ⚙️ 3. Installation (VS Code / Ubuntu)
1️⃣ Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
2️⃣ Install dependencies
```
pip install -r requirements.txt
```
▶️ 4. Run Training
Run from project root:
```
python3 src/train.py

```
This will:

Merge the dataset

Crop faces

Train the model

Save results in models/

Show graphs + confusion matrix
📊 5. Evaluation Metrics

The script automatically prints:

Accuracy curve

Loss curve

Confusion matrix

Precision, Recall, F1

Final Validation Accuracy


📱 6. Use the Model for Prediction
Keras Model Inference
```python
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/deepfake_model.h5")

img = cv2.imread("test.jpg")
img = cv2.resize(img, (224,224))
img = img / 255.0

pred = model.predict(np.expand_dims(img, 0))[0][0]

print("FAKE" if pred > 0.5 else "REAL")
```