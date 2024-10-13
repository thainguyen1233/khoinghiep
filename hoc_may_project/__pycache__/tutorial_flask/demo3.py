from flask import Flask, request, jsonify, send_from_directory
import os
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
import json
import io

use_pretrained = True

model = models.resnet50(pretrained=use_pretrained)
model.eval()

class BaseTransform():
    def __init__(self, resize):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img):
        return self.base_transform(img)

resize = 224 
transform = BaseTransform(resize)

class_index = json.load(open('./imagenet_class_index.json', 'r'))

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index
    
    def predict_max(self, out):
        max_id = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(max_id)]
        return predicted_label_name

predictor = Predictor(class_index)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':
        if 'image' not in request.files and request.data:
            # Xử lý dữ liệu ảnh trực tiếp từ ESP32-CAM
            image_data = request.data
            image = Image.open(io.BytesIO(image_data))
        elif 'image' in request.files:
            image = request.files['image']
        else:
            return jsonify({'error': 'No image data received'}), 400
        
        try:
            img_transformed = transform(image)
            out = model(img_transformed.unsqueeze(0))
            prediction = predictor.predict_max(out)
            return jsonify({
                'message': 'Image processed successfully',
                'prediction': prediction
            }), 200
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=9090)