from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# 使用 MobileNetV2 作為輕量模型
model = models.mobilenet_v2(pretrained=True)
model.eval()

# 類別名稱（可根據實際情境替換）
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# 圖像預處理流程
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        predicted_class = classes[predicted.item()]

    return jsonify({'result': predicted_class})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)