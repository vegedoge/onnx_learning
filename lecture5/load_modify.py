import onnx

# 读取为ModelProto类型的文件
model = onnx.load('linear_func.onnx')
print(model)

# 细节
graph = model.graph
node = graph.node
input = graph.input
output = graph.output

print(node)
print(input)
print(output)

