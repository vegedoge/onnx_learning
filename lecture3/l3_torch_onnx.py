'''
比较TorchScript的trace和script模式的导出
'''

import torch
import torch.onnx

class LoopModel(torch.nn.Module):
    def __init__ (self, n):
        super().__init__()
        self.n = n
        self.conv = torch.nn.Conv2d(3, 3, 3)
        
    def forward(self, x):
        for i in range(self.n):
            x = self.conv(x)
        return x
    
models = [LoopModel(1), LoopModel(2)]
names = ['loop1', 'loop2']

for model, name in zip(models, names):
    dummy_input = torch.randn(1, 3, 12, 12)
    dummy_output = model(dummy_input)
    model_trace = torch.jit.trace(model, dummy_input)
    model_script = torch.jit.script(model)
    
    torch.onnx.export(model_trace, dummy_input, f'{name}_trace.onnx')
    torch.onnx.export(model_script, dummy_input, f'{name}_script.onnx')