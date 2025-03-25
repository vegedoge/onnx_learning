import tensorrt as trt

verbose = True
IN_NAME = "input"
OUT_NAME = "output"
IN_H = 224
IN_W = 224
BATCH_SIZE = 1

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()

with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_builder_config() as config, \
     builder.create_network(EXPLICIT_BATCH) as network:

    # 定义输入张量（显式包含Batch维度）
    input_tensor = network.add_input(
        name=IN_NAME, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H, IN_W)
    )
    # 添加池化层，并直接在构造函数中设置 stride
    pool = network.add_pooling_nd(
        input=input_tensor,
        type=trt.PoolingType.MAX,
        window_size=(2, 2),
    )
    
    pool.get_output(0).name = OUT_NAME
    network.mark_output(pool.get_output(0))

    # 配置优化配置文件
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input=IN_NAME,
        min=(BATCH_SIZE, 3, IN_H, IN_W),
        opt=(BATCH_SIZE, 3, IN_H, IN_W),
        max=(BATCH_SIZE, 3, IN_H, IN_W)
    )
    config.add_optimization_profile(profile)
    config.max_workspace_size = 1 << 30  # 1GB

    # 构建引擎
    engine = builder.build_engine(network, config)
    if engine is None:
        print("Engine构建失败！")
        exit()

    # 保存引擎
    with open("model_python_trt.engine", "wb") as f:
        f.write(engine.serialize())
        print("引擎已保存到 model_python_trt.engine")