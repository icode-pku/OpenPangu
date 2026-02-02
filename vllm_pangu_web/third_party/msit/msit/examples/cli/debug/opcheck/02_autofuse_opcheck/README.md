# 自动融合算子精度预检

## 介绍
该工具能够自动解析ascgraph的图中节点类型和输入输出的dtype，shape等信息，一比一构造一个使用tensorflow的api组成的图，并运行得到一个标杆的结果，进而可以跟ascgraph走融合算子流程时的结果做比对，完成融合算子预检。

## 运行示例
### 1. dump ascgraph和NPU上tensor数据
首先需要确保开启自动融合特性，并配置GE dump graph的相关参数，具体步骤如下：
```sh
# 自动融合开关
export EXPERIMENTAL_ENABLE_AUTOFUSE=1
export EXPERIMENTAL_LOWERING_REDUCE=1
export EXPERIMENTAL_LOWERING_CONCAT=1
# GE dump graph开关
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH=/home/dump_graph  # 配置指定的dump路径
```

NPU的数据dump可以通过Tensorflow Adaptor的ConfigProto进行使能，具体配置如下：
```py
import tensorflow.compat.v1 as tf  # tf1.x版本 修改为 import tensorflow as tf
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

### 2. 执行预检
首先需要source CANN的环境，然后执行预检命令：
```sh
# 示例命令的参数配置都是参考上一节中dump图和tensor数据的路径，需要根据实际情况修改
msit debug opcheck -i {指定到NPU dump数据的上层目录，具体配置提示在‘注意事项’章节中介绍} -m autofuse -gp /home/dump_graph
```
运行结束后，预检结果会保存在执行命令的当前路径下，文件名为 `autofuse_opcheck_result.csv`。

## 注意事项
- 该工具目前仅支持算子类型为AscBackend的融合算子
- 运行过程中工具会在-o参数指定的目录下，创建一个`tmp`的子目录，用于存放NPU dump数据解析后的npy格式的文件，该目录下的文件会在预检结束后自动删除，程序非法退出的话这些文件会被保留用于调试
- -i 参数指定的路径需要是NPU dump数据的上层目录

例如在使能dump的时候，指定路径为/home/dump_data，实际数据会在指定目录下创建子目录进行落盘，具体目录结构为：/home/dump_data/{时间戳}/{device_id}/{模型名}/{模型id}/{step_id}。
因此，-i 参数指定的路径需要是/home/dump_data/{时间戳}/{device_id}/{模型名}/{模型id}/{step_id}，而不是/home/dump_data。