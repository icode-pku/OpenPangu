# GE自动融合整网精度比对场景

## 1. 相关依赖

- CANN（8.1.RC1及以上）
- Tensorflow (1.15.0或是2.6.5)

## 2. 工具安装
需要安装msit工具和compare子工具
- 2.1 安装msit工具，请参考[一体化安装指导](/msit/docs/install/README.md)
- 2.2 安装compare子工具：执行命令： `msit install compare`

## 3. Dump数据 
正式进行数据Dump比对之前，要把模型中涉及到的所有随机性配置全部关闭，包括但不限于对数据集的shuffle，参数的随机初始化等。

### 3.1 dump模型在NPU上推理数据
自动融合特性当前由三个环境变量控制，同时精度比对的算子映射关系依赖于GE的dump图，同样需要三个环境变量进行使能，下面提供一个参考配置示例：
```shell
# 自动融合开关
export EXPERIMENTAL_ENABLE_AUTOFUSE=1
export EXPERIMENTAL_LOWERING_REDUCE=1
export EXPERIMENTAL_LOWERING_CONCAT=1
# GE dump graph开关
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH=/home/dump_graph
```

如果是一个save model格式保存的模型，可以使用msit debug dump工具进行NPU上推理数据的抓取，参考命令如下:
`msit debug dump -m /home/mmoe_model -dp npu -i /home/input_float32.bin -is "input:1,128" -o /home/dump_data/npu`
如果不是save model格式保存的模型，可以通过session配置的方式使能dump，下面提供一个参考示例：
```py
import tensorflow.compat.v1 as tf
config = tf.ConfigProto()

custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["enable_dump"].b = True
custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/dump_data") 
custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0")
custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
with tf.Session(config=config) as sess:
  sess.run() # 模型推理入口
```

### 3.2 dump模型在CPU/GPU上推理数据
可以借助tfdbg或者是session.run函数中fetches参数进行数据dump。需要注意的是，为了保证精度比对结果的有效，需要保证模型输入是相同的。
下面提供一个使用fetches进行dump的示例：
```py
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

input_file = "/home/input_float32.bin"
dump_path = "/home/dump_data/cpu/"
model_path = "/home/frozen_graph.pb"
input_dtype = np.float32
input_shape = [1, 128]

# load model input data
input_data = np.fromfile(input_file, dtype=input_dtype).reshape(input_shape)

# get all node name
node_names = []
data = open(model_path, "rb").read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(data)
for node in graph_def.node:
    node_names.append(node.name)

# session run model
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
    feeds = dict()
    feeds['input:0'] = input_data
    with tf.Session(graph=graph) as sess:
        to_sess_run = [graph.get_tensor_by_name(name + ':0') for name in node_names]
        results = sess.run(to_sess_run, feed_dict=feeds)

        # save dump data
        name_replace = 0
        for idx, node_name in enumerate(node_names):
            tensor = to_sess_run[idx]
            result = results[idx]
            output_node_name = node_name.replace("/", "_") + ".0." + str(int(time.time())) + ".npy"
            save_name = dump_path + output_node_name
            if len(output_node_name) > 210:
                # 落盘文件命令一定要遵循{op_name}.{output_index}.{timestamp}.npy 格式
                replace_file_name = str(name_replace) + ".0." + str(int(time.time())) + ".npy"
                save_name = dump_path + replace_file_name
                print("dump file name: ", output_node_name, " replace to ", str(replace_file_name))
                name_replace += 1
            np.save(save_name, result)
```

上面示例中加载的模型是FrozenGraph格式保存，和Save Model格式的主要区别在于将其中的模型权重变量进行冻结，转为常量，下面是一个模型转换的参考脚本。
```py
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# 加载SavedModel, 当前使用tf2的api
saved_model_dir = '/home/mmoe_model'
loaded_model = tf.saved_model.load(saved_model_dir)
infer = loaded_model.signatures['serving_default']
# 将变量转换为常量，生成冻结的 ConcreteFunction
frozen_func = convert_variables_to_constants_v2(infer)

frozen_graph_def = frozen_func.graph.as_graph_def()
with tf.io.gfile.GFile('frozen_graph.pb', 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())
```

## 4. 执行精度对比
 - 使用ATC工具将dump的GE图转换为json文件，`atc --mode=5 --om={ge_proto_0001_graph_1_build.txt所在路径} --json={转换后文件保存路径}`
 - 执行 `msit debug compare -gp [3.2中数据所在文件] --mp [3.1中数据所在文件] --ops-json [atc转换后json文件路径]`

### 参数说明

| 参数名                 | 描述                                                      | 必选 |
| ---------------------- | ------------------------------------------------------------ | -------- |
| --golden-path, -gp     | CPU/GPU Dump数据根路径                   | 是       |
| --my-path, -mp         | NPU Dump数据根路径                         | 是       |
| --ops-json | 开启自动融合优化后的GE dump图文件 | 是       |
| --output, -o         | 精度比对结果的csv文件保存路径，默认保存在当前路径下         | 否       |