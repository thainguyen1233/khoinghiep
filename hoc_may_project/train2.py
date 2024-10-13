import numpy as np
from ultralytics import YOLO
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

use_pretrained = True
netmodel = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1003)
netmodel.eval()

class BaseTransform():
    def __init__(self, resize):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    def __call__(self, img):
        return self.base_transform(img)

resize = 224
transform = BaseTransform(resize)
image_path = '232b4eb44fbb0a29b2fd5fda51caf8da_tn.jpg'
img = Image.open(image_path)
img_transformed = transform(img)

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        max_id = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(max_id)][1]
        return predicted_label_name

# Load the class index from imagenet_class_index.json
class_index_path = 'imagenet_class_index.json'
with open(class_index_path, 'r') as f:
    class_index = json.load(f)

# Create an instance of Predictor with the loaded class index
predictor = Predictor(class_index)

# Get predictions for the input image
out = netmodel(img_transformed.unsqueeze(0))
result = predictor.predict_max(out)
print("Predicted label:", result)

# Display the transformed image
img_transformed_pil = transforms.ToPILImage()(img_transformed[0])
plt.imshow(np.asarray(img_transformed_pil))
plt.axis('off')
plt.show()
