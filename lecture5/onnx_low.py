import onnx
from onnx import helper
from onnx import TensorProto

# 首先构造张量的原型
a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])

output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])

# 构造算子节点
# 边的信息也被保留在节点里
# if node的输出和某个节点输入匹配，默认节点相连
mul = helper.make_node('Mul', ['a', 'x'], ['c'])
add = helper.make_node('Add', ['c', 'b'], ['output'])

# make graph构造计算图
# 节点必须按照拓扑序给出， 每个节点的输入必须在之前节点的输出里
graph = helper.make_graph(
    [mul, add],
    'linear_func',
    [a, x, b],
    [output]
)

opset_id = helper.make_operatorsetid('', 11) 

# 构造模型
model = helper.make_model(graph, opset_imports=[opset_id])

# check
onnx.checker.check_model(model)
print(model)
onnx.save(model, 'linear_func.onnx')



# test
import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession('linear_func.onnx')
a = np.random.rand(10, 10).astype(np.float32)
x = np.random.rand(10, 10).astype(np.float32)
b = np.random.rand(10, 10).astype(np.float32)

output = sess.run(['output'], {'a': a, 'x': x, 'b': b})[0]

assert np.allclose(output, a * x + b)
print('pass')


