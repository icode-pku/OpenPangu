## 新增特性

v0.2.1版本的新特性包括：

- 支持 dump_data 接口，手动设置 tensor 映射关系实现比对：[自动映射比对使用说明](../v1.0/自动映射比对能力说明.md#第三步手动添加自定义算子映射)
- 支持 torchair GE 图与 FX 图 dump 数据比对
- 支持 dump model 拓扑信息，使用方法：

```
msit llm dump --exec "bash run.sh patches/models/modeling_xxx.py" --type model
```

- 支持dump onnx可视化模型，需要和layer、model配合使用，将dump出来的model和layer拓扑信息，转成onnx可视化模型，使用方法：

```
msit llm dump --exec "bash run.sh patches/models/modeling_xxx.py" --type layer onnx
```

- 支持api方式将之前dump出来的model和layer拓扑信息，转成onnx可视化模型，使用方法：[拓扑信息转onnx可视化模型](#api说明)
- 支持dump torch-npu和torch-gpu模型推理数据，使用方法可参考[接口说明](#api说明)
- 支持opcheck精度预检功能，检测算子精度，使用方法具体参考[精度预检能力使用说明](/msit/docs/llm/工具-精度预检使用说明.md)：

```
msit llm opcheck -i {tensor_dir} -c {op_csv_path} -o {output_dir}
```

## Dump 特性

提供基本的加速库侧算子数据 dump 功能。

### 使用方式

```
msit llm dump --exec "bash run.sh patches/models/modeling_xxx.py"
```

### 参数说明

| 参数名                         | 描述                                                                                                                                                                                                                                                                                         | 必选 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| --exec                         | 指定拉起执行大模型推理脚本的命令，使用示例： --exec "bash run.sh patches/models/modeling_xxx.py"。**注：用户需自行保证执行命令的安全性，并承担因输入不当而导致的任何安全风险或损失；命令中不支持重定向字符，如果需要重定向输出，建议将执行命令写入 shell 脚本，然后启动 shell 脚本**                                                                                 | 是   |
| --type                         | dump类型，可选范围：['model', 'layer', 'op', 'kernel', 'tensor', 'cpu_profiling', 'onnx']，分别表示保存模型拓扑信息、layer拓扑信息、算子信息、kernel算子信息、tensor数据、profiling数据、onnx模型。其中'onnx'需要和'model'、'layer'组合使用。默认为['tensor']。使用方式：--type layer tensor | 否   |
| -sd，--only-save-desc          | 只保存tensor描述信息开关，默认为否。使用方式：-sd                                                                                                                                                                                                                                            | 否   |
| -ids，--save-operation-ids     | 选择dump指定索引的tensor，默认为空，全量dump。使用方式：-ids 24_1,2_3_5                                                                                                                                                                                                                      | 否   |
| -er，--execute-range           | 指定dump的token轮次范围，区间左右全闭，可以支持多个区间序列，默认为第0次，使用方式：-er 1,3 或 -er 3,5,7,7（代表区间[3,5],[7,7],也就是第3，4，5，7次token。）                                                                                                                                | 否   |
| -child，--save-operation-child | 选择是否dump所有子操作的tensor数据，仅使用ids场景下有效，默认为true。使用方式：-child True                                                                                                                                                                                                   | 否   |
| -time，--save-time             | 选择保存的时间节点，取值[0,1,2]，0代表保存执行前(before)，1代表保存执行后(after)，2代表前后都保存。默认保存after。使用方式：-time 0                                                                                                                                                          | 否   |
| -opname，--operation-name      | 指定需要dump的算子类型，支持模糊指定，如selfattention只需要填写self。使用方式：-opname self                                                                                                                                                                                                  | 否   |
| -tiling，--save-tiling         | 选择是否需要保存tiling数据，默认为false。使用方式：-tiling                                                                                                                                                                                                                                   | 否   |
| --save-tensor-part, -stp       | 指定保存tensor的部分，0为仅intensor，1为仅outtensor，2为全部保存，默认为2。使用示例：-stp 1                                                                                                                                                                                                  | 否   |
| -o, --output                   | 指定dump数据的输出目录，默认为'./'，使用示例：-o aasx/sss                                                                                                                                                                                                                                    | 否   |
| -h, --help            | 命令行参数帮助信息 | 否    |
### Dump 落盘位置

Dump默认落盘路径 `{DUMP_DIR}`在当前目录下，如果指定output目录，落盘路径则为指定的 `{OUTPUT_DIR}`。

- tensor 信息会生成在默认落盘路径的 ait_dump 目录下，具体路径是 `{DUMP_DIR}/ait_dump/tensors/{device_id}_{PID}/{TID}`目录下。
- layer 信息会生成在默认落盘路径的 ait_dump 目录下，具体路径是 `{DUMP_DIR}/ait_dump/layer/{PID}`目录下。
- model 信息会生成在默认落盘路径的 ait_dump 目录下，具体路径是 `{DUMP_DIR}/ait_dump/model/{PID}`目录下。注：由于 model 由 layer 组合而成，因此使用 model 时，默认同时会落盘 layer 信息。
- onnx 需要和 layer、model 配合使用，落盘位置和 model、layer 相同的目录。
- cpu_profiling 信息会生成在默认落盘路径的 ait_dump 目录下，具体路径是 `{DUMP_DIR}/ait_dump/cpu_profiling/{TIMESTAMP}/operation_statistic_{executeCount}.txt`。
- 算子信息会生成在默认落盘路径的 ait_dump 目录下，具体路径是 `{DUMP_DIR}/ait_dump/operation_io_tensors/{PID}/operation_tensors_{executeCount}.csv`。
- kernel 算子信息会生成在默认落盘路径的 ait_dump 目录下，具体路径是 `{DUMP_DIR}/ait_dump/kernel_io_tensors/{PID}/kernel_tensors_{executeCount}.csv`。

注：`{device_id}`为设备号；`{PID}`为进程号；`{TID}`为 `token_id`；`{TIMESTAMP}`为时间戳；`{executeCount}`为 `operation`运行次数。

---

### API说明

#### 拓扑信息转onnx可视化模型：

```python
from llm.common.json_fitter import atb_json_to_onnx

model_level = 1   # 可视化模型的节点深度，按需填写，比如填写为1，则表示生成深度为1的可视化模型，不填默认生成最大深度可视化模型
layer_topo_info = "./XXX_layer.json"   # dump出来的layer拓扑信息或者model拓扑信息
atb_json_to_onnx(layer_topo_info, model_level)
```

#### dump torch-npu(gpu)模型推理数据

##### DumpConfig

接口说明：dump数据配置类，可用于按需dump模型数据。

接口原型：DumpConfig(dump_path, token_range, module_list, tensor_part)

| 参数名      | 含义                   | 使用说明                                                                                                                                 | 是否必填 |
| ----------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| dump_path   | 设置dump的数据路径     | 数据类型：str，默认为当前目录。                                                                                                          | 否       |
| token_range | 需要dump的token列表    | 数据类型：list。默认为[0]，只dump第0个token的数据。                                                                                      | 否       |
| module_list | 指定要hook的module类型 | 数据类型：list，默认为[]，即dump所有module的数据。                                                                                       | 否       |
| tensor_part | 指定要dump哪部分数据   | 数据类型：int，默认为2。当tensor_part=0时，只dump输入数据；当tensor_part=1时，只dump输出数据； 当tensor_part=2时，dump输入和输出的数据。 | 否       |

##### register_hook

接口说明：给模型添加hook，用于dump数据

接口原型：register_hook(model, config, hook_type=”dump_data”)

| 参数名    | 含义           | 使用说明                                                | 是否必填 |
| --------- | -------------- | ------------------------------------------------------- | -------- |
| model     | 需要hook的模型 | 数据类型：torch.nn.Module，建议设置为最外层的torch模型  | 是       |
| config    | Hook配置       | 数据类型：DumpConfig                                    | 是       |
| hook_type | hook类型       | 数据类型：str，默认值为dump_data，当前仅支持dump_data。 | 否       |

##### 使用示例

```
from llm import DumpConfig, register_hook
dump_config = DumpConfig(dump_path="./ait_dump")
register_hook(model, dump_config)  # model是要dump中间tensor的模型实例，在模型初始化后添加代码
```

## Compare 特性

提供有精度问题的数据与标杆数据之间的比对能力。

### 使用方式

```
msit llm compare --golden-path golden_data.bin --my-path my-path.bin
```

#### 参数说明

| 参数名             | 描述                                       | 是否必选 |
| ------------------ | ------------------------------------------ | -------- |
| --golden-path, -gp | 标杆数据路径，支持单个数据文件路径或文件夹 | 是       |
| --my-path, -mp     | 待比较的数据路径，为单个数据文件路径       | 是       |
| --log-level, -l    | 日志级别，默认为info                       | 否       |
| --output, -o       | 比对结果csv的输出路径                      | 否       |

## Opcheck 特性

支持算子精度预检，根据dump出的tensor及算子信息，执行单算子UT，检测算子精度。具体参考[精度预检能力使用说明](/msit/docs/llm/工具-精度预检使用说明.md)。

### 使用方式

```
msit llm opcheck -i {tensor_dir} -c {op_csv_path} -o {output_dir}
```

#### 参数说明

| 参数名                      | 描述                                                                                                                                                                    | 是否必选 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| --input, -i                 | tensor数据路径，为文件夹，由msit llm dump --type tensor落盘，示例：OUTPUT_DIR/{device_id}_{PID}/{TID}/                                                                                 | 是       |
| --csv-path, -c              | 算子信息csv文件路径，为单个数据文件路径，由msit llm dump --type op落盘，示例：OUTPUT_DIR/ait_dump/operation_io_tensors/PID/operation_tensors_0.csv                       | 是       |
| --output, -o                | 输出文件的保存路径，为文件夹，示例：xx/xxx/xx                                                                                                                           | 否       |
| --operation-ids, -ids       | 选择预检指定索引的tensor，默认为空，全量算子预检。使用方式：-ids 24_1,2_3_5                                                                                             | 否       |
| --operation-name, -opname   | 指定需要预检的算子类型，支持模糊指定，如selfattention只需要填写self。使用方式：-opname self，linear                                                                     | 否       |
| --precision-metric, -metric | 指定需要输出的精度类型，可选范围：['abs', 'cos_sim'，'kl']，分别表示绝对误差通过率、余弦相似度、KL散度。默认为[]，即只输出相对误差通过率。使用方式：--metric kl cos_sim | 否       |
| --device-id, -device        | 指定需要使用的NPU设备，默认为0                                                                                                                                          | 否       |
***

## TorchAir 图模式 GE dump 数据与 FX dump 数据精度比对
### Dump 数据
- **GE 模式 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理
  ```py
  import torch, torch_npu, torchair
  from llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```
  输出路径为指定的 `{dump_path}/dump_{time_stamp}`
- **FX 模式 dump 数据** 添加 `get_fx_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理
  ```py
  import torch, torch_npu, torchair
  from llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_fx_dump_config()  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```
  输出路径为当前文件夹下的 `gm_{time stamp}_dump`
### Compare 比对
  - 执行 `msit llm compare --my-path [GE dump data] --golden-path [FX dump data]`，输出比对结果 csv 文件
    ```sh
    msit llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path gm_{time stamp}_dump
    ```
***

## TorchAir 图模式融合的 GE dump 数据与关闭融合的 GE dump 数据精度比对
### Dump 数据
- **GE 模式 dump 数据** 添加 `get_ge_dump_config`，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理
  ```py
  import torch, torch_npu, torchair
  from llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```
  输出路径为指定的 `{dump_path}/dump_{time_stamp}`
- **GE 模式关闭融合 dump 数据** 添加 `get_ge_dump_config`，指定 `fusion_switch_file` 文件，获取配置后的 `CompilerConfig` 实例，配置模型 compile，并执行推理
  ```py
  import torch, torch_npu, torchair
  from llm.dump import torchair_dump  # 添加导入
  ...
  model = ...
  config = torchair_dump.get_ge_dump_config(dump_path="dump", fusion_switch_file="fusion_switch.json")  # 添加获取 config
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  ...
  ```
  参考的 `fusion_switch.json` 文件
  ```json
  {
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
      },
      "UBFusion": {
        "ALL": "off"
      }
    }
  }
  ```
  输出路径为指定的 `{dump_path}/dump_{time_stamp}`
### Compare 比对
- 执行 `msit llm compare --my-path [GE dump data] --golden-path [fusion off GE dump data]`，输出比对结果 csv 文件
    ```sh
    msit llm compare --my-path {dump_path}/dump_{time_stamp} --golden-path {dump_path}/dump_{time_stamp}
    ```