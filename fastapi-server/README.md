### 🚗 Car Damage Classification API
### 📖 Project Description

This project is a Car Damage Classification API built using FastAPI and PyTorch.
It uses a pretrained ResNet50 deep learning model to classify car damage images into 6 categories.

### ✨ Features

1. Uses ResNet50 (Pretrained) model

2. Applies Transfer Learning

3. Freezes all layers

4. Unfreezes layer4

5. Replaces and trains fc layer

6. FastAPI REST API

7. Upload image → Get prediction

8. CPU compatible

### 🏷️ Damage Categories

1. Front Breakage

2. Front Crushed

3. Front Normal
4. Rear Breakage

5. Rear Crushed

6. Rear Normal

### 🧠 Model Architecture
🔹 Base Model

ResNet50 (Pretrained on ImageNet)

🔹 Input Details

Image Size: 224 × 224

Color Mode: RGB

Normalization:

Mean: [0.485, 0.456, 0.406]

Std: [0.229, 0.224, 0.225]

🔹 Training Strategy

All layers frozen

Only:

     layer4

     fully connected (fc) layer
     are trained

### ⚙️ Installation
1️⃣ Create Virtual Environment
python -m venv venv
Activate Environment

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
2️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run the Server
uvicorn server:app --reload

Server runs at:

http://127.0.0.1:8000
🌐 API Usage
🔹 Endpoint

POST /predict

🔹 Swagger Documentation

#### Open in browser:

http://127.0.0.1:8000/docs
🔹 Using cURL
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@car.jpg"
### 📥 Example Response
{
  "prediction": "Front Crushed"
}

### 📦 Requirements

Python 3.9+

fastapi==0.134.0

torch==2.10.0

torchvision==0.25.0

Pillow==11.1.0

python-multipart==0.0.22

### 🚀 How It Works

1. User uploads an image

1. Image is resized to 224×224

3. Image is normalized (ImageNet standards)

4. Model predicts damage class

5. API returns JSON response