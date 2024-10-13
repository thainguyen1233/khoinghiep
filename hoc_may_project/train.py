
'''tiền xử lý hậu xử lý , in/out của mạng
12 tạo dataset
3. tạo dataloader
4.xây dựng netwwork
5. định nghĩa hàm forward(). trong networlk
6.định nghĩa hàm loss
7. thiết lập thuật toán tối ưu
8. thiết lập việc học và kiểm thử network
8. dự đoán với data test
'''
import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
# sử dụng thư viện pytorch để nhận diện và xử lý ảnh 
# dùng import để tải thư viện vgg16
use_pretrained = True
net = models.vgg19(pretrained=use_pretrained)
net.eval()# sử dụng mô hình vgg16 đã được huán luyện trước, dùng hàm eval để đánh giá mô hình đó
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
image_path = 'chainhua.jpg'
img = Image.open(image_path)
img_transformed = transform(img)

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out): 
        max_id = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(max_id)]
        return predicted_label_name

class_index = json.load(open('./imagenet_class_index.json', 'r'))
predictor = Predictor(class_index)
out = net(img_transformed.unsqueeze(0)) 
result = predictor.predict_max(out)
print("Kết quả là:", result)
img_transformed_pil = transforms.ToPILImage()(img_transformed)
plt.imshow(np.asarray(img_transformed_pil))
plt.axis('off')
plt.show()
