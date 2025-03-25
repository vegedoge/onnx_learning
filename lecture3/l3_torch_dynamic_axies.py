import torch
import onnxruntime as ort
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
model = Model()
dummy_input = torch.randn(1, 3, 10, 10)
model_names = ['model.onnx', 'model_dynamic_0.onnx', 'model_dynamic_1.onnx']

dynamic_axes_0 = {
    'in': {0: 'batch'},
    'out': {0: 'batch'}
}

dynamic_axes_23 = {
    'in': {2: 'height', 3: 'width'},
    'out': {2: 'height', 3: 'width'}
}

torch.onnx.export(model, dummy_input, model_names[0],
                  input_names=['in'], output_names=['out'])

torch.onnx.export(model, dummy_input, model_names[1],
                    input_names=['in'], output_names=['out'],
                    dynamic_axes=dynamic_axes_0)

torch.onnx.export(model, dummy_input, model_names[2],
                    input_names=['in'], output_names=['out'],
                    dynamic_axes=dynamic_axes_23)

# test
origin_tensor = np.random.rand(1,3,10,10).astype(np.float32)
mult_batch_tensor = np.random.rand(2,3,10,10).astype(np.float32)
big_tensor = np.random.rand(1,3,20,20).astype(np.float32)

inputs = [origin_tensor, mult_batch_tensor, big_tensor]
exceptions = dict()

for model_name in model_names:
    for i, input in enumerate(inputs):
        try:
            sess = ort.InferenceSession(model_name)
            ort_inputs = {'in': input}
            sess.run(['out'], ort_inputs)
        except Exception as e:
            exceptions[(i, model_name)] = e
            print(f'{model_name} run {i} failed')
        else:
            print(f'{model_name} run {i} success')

