# 加速库模型数据 dump

提供大模型推理过程中产生的中间数据的 dump 能力，包括：

1. **【Dump Tensor 能力】** 用于保存 单 operation 或 图 operation 的输入输出及中间张量，主要精度比对时使用
2. **【保存拓扑信息能力】** 包含两个维度的拓扑信息的保存能力，分别为 operation 维度和模型维度，operation 维度的拓扑信息保存能力主要是保存 ATB 的 operation 的拓扑信息，用于图结构分析；模型维度拓扑信息保存能力主要依赖于MindIE-LLM，用于保存MindIE-LLM内 model 的拓扑信息，进行模型结构分析，或自动精度比对，自动比对需要先知道模型拓扑信息。模型拓扑信息可以转换成 onnx，可以使用可视化工具打开查看
3. **【保存 operation 信息能力】** 保存 ATB operation 的多项属性，如参数、输入张量 Shape、输出张量 Shape 等
4. **【保存 kernel 信息能力】** 保存 kernel operation 的多项属性，是比 ATB operation 更细粒度，是 ATB operation 的组成部分，多数是算子开发人员定位使用
5. **【保存 cpu_profiling 数据能力】** 保存 cpu profiling 信息，主要用于 host 侧性能定位，数据下发慢等问题，主要是算子开发、熟悉 atb 框架的开发人员定位使用
6. **【保存 tiling 数据能力】** tiling 数据是 host 侧计算生成，用于 device 侧进行数据切分。主要用于算子开发人员定位算子精度异常问题

- 加速库算子 tensor 及拓扑信息保存会占用磁盘，当落盘路径磁盘空间小于2G，会输出如下提示："Disk space is not enough, it's must more than 2G, free size(MB) is:"

## 使用方式

```bash
msit llm dump --exec "<任意包含ATB的程序执行命令>" [可选参数]

# dump 不同类型数据
msit llm dump --exec "<任意包含ATB的程序执行命令>" --type model tensor # 常用用于自动比对
msit llm dump --exec "<任意包含ATB的程序执行命令>" --type onnx # 常用于导出onnx查看网络结构

# 仅dump统计量
msit llm dump --exec "<任意包含ATB的程序执行命令>" --type model tensor stats # 查看模型tensor的统计量, 相比全量落盘，可节省磁盘空间，但需花费额外时间进行统计量的计算

# 仅dump layer 层的算子输出，常用于精度比对先找到存在问题的 layer 层。相比全量dump，可以节省磁盘空间和定位时间
msit llm dump --exec "<任意包含ATB的程序执行命令>" --type model tensor -child False

# dump 不同轮次数据
msit llm dump --exec "<任意包含ATB的程序执行命令>" --type model tensor -er 1,1 # dump输出轮次为 1 的数据，根据实际情况指定。需要考虑是否有warmup，是否有prefill

# dump 指定算子
msit llm dump --exec "<任意包含ATB的程序执行命令>" --type model tensor -ids 3 # dump 编号为 3 的layer的输入输出数据
msit llm dump --exec "<任意包含ATB的程序执行命令>" --type model tensor -ids 3_1 # dump 编号为 3 的layer数据中第 1 个子算子的输入输出数据

# dump 时开启/关闭 用软链接存储落盘文件
msit llm dump --exec "<任意包含ATB的程序执行命令>" # 默认为关闭
msit llm dump --exec "<任意包含ATB的程序执行命令>" --enable-symlink # 打开软链接存储落盘文件的功能
```

## 命令行参数

| 参数名                           | 参数描述                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 是否必选 |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ---- |
| --exec                        | 指定包含 ATB 的程序执行命令，使用示例： --exec "bash run.sh patches/models/modeling_xxx.py"。**注：用户需自行保证执行命令的安全性，并承担因输入不当而导致的任何安全风险或损失；命令中不支持重定向字符，如果需要重定向输出，建议将执行命令写入 shell 脚本，然后启动 shell 脚本**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 是   |
| --type                        | dump 类型，默认为['tensor', 'model']。使用方式：--type layer tensor。可选项有：<br /> model: 模型拓扑信息(默认)，当dump model的时候，layer会跟着model一起dump下来<br /> layer: operation 维度拓扑信息<br /> op: ATB operation 信息<br /> kernel: kernel operation 信息<br /> tensor: tensor 数据(默认)<br /> stats: 必须在--type同时填选tensor, 即[--type 其它 tensor stats], 会根据tensor的数据(仅支持数值数据类型，不支持bool/string等不可计算的数据类型)来计算统计量: [format、type、dims、max、min、mean、l2norm], **注：1. 同时选择tensor和stats时，不再落盘全量tensor数据，仅落盘以上所述7种统计量; 2. 单个tensor中所有元素的平方和不得超过C++中`double`支持的范围，即平方和最大值为(1.79769 * 10 ^ 308)，否则将在计算时出现数据溢出从而导致结果与预期不符** <br />  cpu_profiling: cpu profiling 数据<br /> onnx: onnx 模型。仅用于模型结构可视化 | 否   |
| -sd，--only-save-desc          | 只保存 tensor 描述信息开关，默认为否，开启开关时将 dump tensor 的描述信息，使用方式：-sd                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 否   |
| -ids，--save-operation-ids     | 设置 dump 指定 id 的算子的 tensor，默认为空，全量 dump。使用方式：-ids 2, 3_1 表示只 dump 第 2 个 operation 和第 3 个 operation 的第 1 个算子的数据，id 从 0 开始。若不确定算子 id，可以先执行 msit llm dump --exec xx --type model 命令，将 model 信息 dump 下来，即可获得模型中所有的算子 id 信息。                                                                                                                                                                                                                                                                                                                                                                                                                        | 否   |
| -er，--execute-range           | 指定 dump 模型推理的执行轮次范围，区间左右全闭，可以支持多个区间序列，默认为第 0 次，使用方式：-er 1,3 （下标从0开始计数，代表1~3轮的推理执行）或 -er 3,5,7,7（代表区间[3,5],[7,7],也就是第 3，4，5，7 轮的推理执行）。此外，请确保输入多区间时的总输入长度不超过500个字符。                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 否   |
| -child，--save-operation-child | 选择是否 dump 所有子操作的 tensor 数据，仅使用 ids 场景下有效，默认为 true。使用方式：-child True                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 否   |
| -time，--save-time             | 选择保存的时间节点，取值[0,1,2,3]，0 代表保存执行前(before)，1 代表保存执行后(after)，2 代表前后都保存(both), 3 代表保存执行前的intensor和执行后的outtensor。默认值为 3。使用方式：-time 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 否   |
| -opname，--operation-name      | 指定需要 dump 的算子类型，只需要指定算子名称的开头，可以模糊匹配，如 selfattention 只需要填写 self。使用方式：-opname self                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 否   |
| -tiling，--save-tiling         | 选择是否需要保存 tiling 数据，默认为 false。使用方式：-tiling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 否   |
| --save-tensor-part, -stp      | 指定保存 tensor 的部分，0 为仅 intensor，1 为仅 outtensor，2 为全部保存，默认为 2。使用示例：-stp 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 否   |
| -o, --output                  | 指定 dump 数据的输出目录，默认为'./'，使用示例：-o aasx/sss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 否   |
| -device, --device-id          | 指定 dump 数据的 device id，默认为 None 表示不限制。如指定 --device-id 1，将只 dump 1 卡的数据                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 否   |
| -symlink, --enable-symlink    | 选择正在 dump 的数据若与已保存的文件中的数据相同时， 是否使用软链接来节省磁盘空间和运行时间，默认为不开启。如指定 --enable-symlink 或 -symlink，则开启使用软链接功能                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 否   |
| -config, --config-path    | 指定 dump 配置文件。指定配置文件后，可在模型执行过程中，进行 dump 参数的修改，实现动态 dump。配置文件具体介绍参见“[Dump 配置文件](#Dump-配置文件)”章节。默认不使用 dump 配置文件。本参数仅支持 8.3 及以上 CANN 版本环境                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 否   |
| -l, --log-level               | 指定 log level，默认为 info，可选值 debug, info, warning, error, fatal, critical                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 否   |
| -h, --help                    | 命令行参数帮助信息                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 否 | 
| -seed                | 设定确定性计算的种子，默认为None表示不开启确定性计算。如果设置种子，可以不输入。如果不输入那么为2024，如果输入那么需要输入一个能转换为int的值。使用示例：-seed 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 否   |

## Dump 配置文件

Dump 配置文件为 JSON 格式文本文件。指定配置文件后，命令行参数中的 "-ids", "-er", "-child", "-device" 不再生效，相关 Dump 行为仅受配置文件控制。Dump 配置文件中的参数介绍如下：

| 参数名 | 参数描述 | 是否必选 |
|-------|---------|----------|
| dump_enable | 指定是否开启数据dump数据。str类型。只有为"true"时，才开启dump。""同"false" | 否(默认为"false") |
| er          | 含义同命令行"-er"参数。"all"表示dump所有执行轮次的数据，""同"0,0"         | 否(默认为"0,0")   |
| ids         | 含义同命令行"-ids"参数。""表示dump任意id的算子的数据                      | 否(默认为"")      | 
| child       | 含义同命令行"-child"参数。""同"true"                                    | 否(默认为"true")  | 
| device      | 含义同命令行"-device"参数。""表示dump所有device的数据                    | 否(默认为"")      | 

Dump 配置文件示例如下。使用示例配置可 dump 所有 device 上 id 为 0、1、2 的 operation 及其子 operation 的第一次执行的精度数据。

```json
{
    "dump_enable": "true",
    "er": "0,0",
    "ids": "0,1,2",
    "child": "true",
    "device": ""
}
```

**注意**：配置文件中的参数修改需至少在 token 执行前 5 秒完成，才可保证生效。

## 结果查看

### Dump 落盘位置

Dump 默认落盘路径 `{DUMP_DIR}`在当前目录下，如果指定 output 目录，落盘路径则为指定的 `{OUTPUT_DIR}`。

注：`{device_id}`为设备号；`{PID}`为进程号；`{TID}`为 `token_id`；`{TIMESTAMP}`为时间戳；`{executeCount}`为 `operation`运行次数。

- tensor 信息，具体路径是 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/tensors/{device_id}_{PID}/{TID}`目录下（使用老版本的 cann 包可能导致 tensor 落盘路径不同）。
- stats 统计量信息，具体路径是 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/tensors/{device_id}_{PID}/{TID}`目录下（同`tensor 信息`落盘位置）。
- layer 信息，具体路径是 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/layer/{PID}`目录下。
- model 信息，具体路径是 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/model/{PID}`目录下。注：由于 model 由 layer 组合而成，因此使用 model 时，默认同时会落盘 layer 信息。
- onnx 落盘位置和 model、layer 相同的目录。（落盘onnx文件格式为 xxx.onnx）
- cpu profiling 信息，具体路径是 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/cpu_profiling/{PID}/operation_statistic_{executeCount}.csv`。
- 算子信息，具体路径是 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/operation_io_tensors/{PID}/operation_tensors_{executeCount}.csv`。
- kernel 算子信息，具体路径是 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/kernel_io_tensors/{PID}/kernel_tensors_{executeCount}.csv`。

##### 模型拓扑信息转 onnx 可视化模型：

```python
from msit_llm.common.json_fitter import atb_json_to_onnx

model_level = 1   # 可视化模型的节点深度，按需填写，比如填写为1，则表示生成深度为1的可视化模型，不填默认生成最大深度可视化模型
layer_topo_info = "./XXX_layer.json"   # dump出来的layer拓扑信息或者model拓扑信息
atb_json_to_onnx(layer_topo_info, model_level, {})
```

* 参数说明  
model_level：可视化模型的节点深度，按需填写，比如填写为1，则表示生成深度为1的可视化模型，不填默认生成最大深度可视化模型  
layer_topo_info：dump出来的layer拓扑信息或者model拓扑信息，json格式的文件    
atb_json_to_onnx：读取op信息的缓存，多次调用可以复用缓存信息。外部调用可以不用关注，直接传入空的dict即可

### 典型场景Dump数据说明
通常情况下，完整的推理过程，可以分为一次prefill+多次decode。同时现阶段模型大部分都存在warmup阶段，因此下面举例说明不同场景下的dump数据含义：
1. 模型推理无warmup阶段
```lua
├── msit_dump_{TIMESTAMP}
│   ├── tensors
│   |   ├── {device_id}_{PID}
│   |   │   ├──  0                                               #模型执行第一轮的tensor数据
|   |   |   |    ├── 0_WordEmbedding/                                  #词嵌入操作tensor数据
|   |   |   |    ├── 1_PositionalEmbeddingGather/                      #位置编码操作tensor数据
|   |   |   |    ├── 2_Prefill_layer/                                  #prefill阶段数据
|   |   |   |    ├── 3_Prefill_layer/    
|   |   |   |    ...
|   |   |   |    ├── 40_Decoder_layer/                                 #decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
│   |   │   ├──  1                                                     #模型执行第二轮的tensor数据
|   |   |   |    ├── 40_Decoder_layer/                                 #由于prefill在推理阶段只进行一次，该轮仅存在decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
│   |   │   ├──  ...
|   |   |   ├──  n
|   |   |   |    ├── 40_Decoder_layer/                                 #decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
```
2. 模型推理有warmup阶段（warmup阶段仅做prefill）
```lua
├── msit_dump_{TIMESTAMP}
│   ├── tensors
│   |   ├── {device_id}_{PID}
│   |   │   ├──  0                                               #模型执行第一轮（warmup）的tensor数据
|   |   |   |    ├── 0_WordEmbedding/                                  #词嵌入操作tensor数据
|   |   |   |    ├── 1_PositionalEmbeddingGather/                      #位置编码操作tensor数据
|   |   |   |    ├── 2_Prefill_layer/                                  #prefill阶段数据
|   |   |   |    ├── 3_Prefill_layer/    
|   |   |   |    ...
│   |   │   ├──  1                                               #模型执行第二轮（正式推理）的tensor数据
|   |   |   |    ├── 0_WordEmbedding/                                  #词嵌入操作tensor数据
|   |   |   |    ├── 1_PositionalEmbeddingGather/                      #位置编码操作tensor数据
|   |   |   |    ├── 2_Prefill_layer/                                  #prefill阶段数据
|   |   |   |    ├── 3_Prefill_layer/    
|   |   |   |    ...
|   |   |   |    ├── 40_Decoder_layer/                                 #decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
│   |   │   ├──  2                                               #模型执行第三轮的tensor数据
|   |   |   |    ├── 40_Decoder_layer/                                 #由于prefill在推理阶段只进行一次，该轮仅存在decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
│   |   │   ├──  ...
|   |   |   ├──  n                                               #模型执行第n轮的tensor数据
|   |   |   |    ├── 40_Decoder_layer/                                 #decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
```
3. 模型推理有warmup阶段（warmup阶段做prefill+decode）
```lua
├── msit_dump_{TIMESTAMP}
│   ├── tensors
│   |   ├── {device_id}_{PID}
│   |   │   ├──  0                                               #模型执行第一轮（warmup）的tensor数据
|   |   |   |    ├── 0_WordEmbedding/                                  #词嵌入操作tensor数据
|   |   |   |    ├── 1_PositionalEmbeddingGather/                      #位置编码操作tensor数据
|   |   |   |    ├── 2_Prefill_layer/                                  #prefill阶段数据
|   |   |   |    ├── 3_Prefill_layer/    
|   |   |   |    ...
|   |   |   |    ├── 40_Decoder_layer/                                 #decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
│   |   │   ├──  1                                               #模型执行第二轮（正式推理）的tensor数据
|   |   |   |    ├── 0_WordEmbedding/                                  #词嵌入操作tensor数据
|   |   |   |    ├── 1_PositionalEmbeddingGather/                      #位置编码操作tensor数据
|   |   |   |    ├── 2_Prefill_layer/                                  #prefill阶段数据
|   |   |   |    ├── 3_Prefill_layer/    
|   |   |   |    ...
|   |   |   |    ├── 40_Decoder_layer/                                 #decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
│   |   │   ├──  2                                               #模型执行第三轮的tensor数据
|   |   |   |    ├── 40_Decoder_layer/                                 #由于prefill在推理阶段只进行一次，该轮仅存在decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
│   |   │   ├──  ...
|   |   |   ├──  n                                               #模型执行第n轮的tensor数据
|   |   |   |    ├── 40_Decoder_layer/                                 #decode阶段数据
|   |   |   |    ├── 41_Decoder_layer/  
|   |   |   |    ...
```
### 查看和Dump数据

读取、转换和保存 bin 数据的接口可以参考[API-读取和保存接口](./API-读取和保存接口.md)