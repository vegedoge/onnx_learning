import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.asinh(x)
    
from torch.onnx import register_custom_op_symbolic

def asinh_symbolic(g, input, *, out=None):
    return g.op('Asinh', input)

register_custom_op_symbolic('aten::asinh', asinh_symbolic, 9)

model = Model()

input = torch.rand(1,3,10,10)
torch.onnx.export(model, input, 'asinh.onnx', opset_version=9, input_names=['in'], output_names=['out'])

import numpy as np
import onnxruntime as ort

torch_output = model(input).detach().numpy()

sess = ort.InferenceSession('asinh.onnx')
ort_output = sess.run(['out'], {'in': input.numpy()})

assert np.allclose(torch_output, ort_output[0], atol=1e-6)
