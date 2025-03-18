import os

import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn
from torch.nn.functional import interpolate
import onnxruntime as ort
from interpolate_op import NewInterpolate

class SuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU()
        
    def forward(self, x, upscale_factor):
        x = NewInterpolate.apply(x,
                                 upscale_factor)
        
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
    
# download checkpoint and test image
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth', 
    'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png'] 
names = ['srcnn.pth', 'face.png'] 
for url, name in zip(urls, names): 
    if not os.path.exists(name): 
        open(name, 'wb').write(requests.get(url).content) 
        
def init_torch_model():
    torch_model = SuperResolutionNet()

    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:]) 
        state_dict[new_key] = state_dict.pop(old_key)
        
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model
        
model = init_torch_model()
input_img = cv2.imread('face.png').astype(np.float32)

factor = torch.tensor([1,1,3,3], dtype=torch.float32)

# HWC to NCHW
input_img = np.transpose(input_img, (2, 0, 1))
input_img = np.expand_dims(input_img, axis=0)

# inference
torch_output = model(torch.tensor(input_img), factor).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, (1, 2, 0)).astype(np.uint8)



x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    torch.onnx.export(
        model,
        (x,factor),
        "srcnn_op.onnx",
        opset_version=11,
        input_names=["input", "factor"],
        output_names=["output"],
    )

# dynamic factor with self defined resize op
input_factor = np.array([1,1,4,4], dtype = np.float32)

# 获取推理器
ort_session = ort.InferenceSession("srcnn_op.onnx")
ort_inputs = {'input': input_img, 'factor': input_factor}

# 第一个参数是输出张量列表， 第二个为输入张量字典
# 对应导出时候的输入输出名称
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = ort_output.transpose(1, 2, 0).astype(np.uint8)
cv2.imwrite('face_ort_op.png', ort_output)

# show
cv2.imwrite('face_torch_op.png', torch_output)
