# msit debug dump功能使用指南
## 简介
- 提供了传统小模型场景下tensor数据dump功能，辅助进行模型定位debug。适用于 TensorFlow、TensorFlow2.0、ONNX、Caffe、MindIE-Torch模型，用户只需要通过参数指定原始模型对应的离线模型和模型输入文件。模型输入的 bin 文件需要符合模型的输入要求（支持模型多输入）。并且额外提供了dump TensorFlow1.x框架下.pb模型数据的工具TFDebugRunner，使用方法参见[TFDebugRunner使用说明](tf_debug_runner.md)。


## 工具安装
- 安装命令：
- **注:** 目前debug dump 的安装集成到了 debug compare 命令里，用户只需要安装debug compare即可实现debug dump功能，命令如下：
```shell
msit install compare
```

## 使用方法
### 功能介绍
#### 使用入口
dump功能可以直接通过msit命令行形式启动精度对比。启动方式如下：

**不指定模型输入** 命令示例，**其中路径需使用绝对路径**
  ```sh
  msit debug dump -m /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -dp cpu
  -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test
  ```

### 输出结果说明

```sh
{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape} 用来区分动态shape时不同的模型实际输入，静态shape时没有该层
├-- dump_data
│   ├-- npu                          # npu dump 数据目录
│   │   ├-- {timestamp}              # 模型所有npu dump的算子输出
│   │   │   └-- 0                    # Device 设备 ID 号
│   │   │       └-- {om_model_name}  # 模型名称
│   │   │           └-- 1            # 模型 ID 号
│   │   │               ├-- 0        # 针对每个Task ID执行的次数维护一个序号，从0开始计数，该Task每dump一次数据，序号递增1
│   │   │               │   ├-- Add.8.5.1682067845380164
│   │   │               │   ├-- ...
│   │   │               │   └-- Transpose.4.1682148295048447
│   │   │               └-- 1
│   │   │                   ├-- Add.11.4.1682148323212422
│   │   │                   ├-- ...
│   │   │                   └-- Transpose.4.1682148327390978
│   │   ├-- {time_stamp}
│   │   │   ├-- output_0.bin
│   │   │   └-- output_0.npy
│   │   └-- {time_stamp}_summary.json
│   └-- {onnx or tf or caffe} # 原模型 dump 数据存放路径，onnx / tf / caffe 分别对应 ONNX / Tensorflow / Caffe 模型
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # 随机输入数据，若指定了输入数据，则该文件不存在
├-- model
│   ├-- {om_model_name}.json                    # 离线模型om模型(.om)通过atc工具转换后的json文件
│   └-- new_{onnx_model_name}.onnx              # 把每个算子作为输出节点后新生成的 onnx 模型
└-- tmp                                      # 如果 -m 模型为 Tensorflow pb 文件, tfdbg 相关的临时目录
```

#### 安全风险提示

在模型文件传给工具加载前，用户需要确保传入文件是安全可信的，若模型文件来源官方有提供SHA256等校验值，用户必须要进行校验，以确保模型文件没有被篡改。

#### 命令行入参说明

| 参数名                       | 描述                                                                                                                                                                                                                                                                                | 必选 |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----|
| -m，--model           | 模型文件 [.pb与saved_model，.onnx，.prototxt] 路径，分别对应 TF, ONNX, Caffe。<br/>其中.pb为TF1.15.5版本模型文件，saved_model为TF2.6.5版本模型文件                                                                                                                                                                | 否  |
| -w，--weight               | -w 为权重文件，当模型为caffe模型时，该参数为必选参数                                                                                                                                                                                                                                                    | 否  |
| -i，--input                | 模型的输入数据路径，默认根据模型的input随机生成，多个输入以逗号分隔，例如：/home/input\_0.bin,/home/input\_1.bin,/home/input\_2.npy。注意：使用aipp模型时该输入为om模型的输入,且支持自动将npy文件转为bin文件                                                                                                                                       | 否  |
| -c，--cann-path            | CANN包安装完后路径，默认会从系统环境变量`ASCEND_TOOLKIT_HOME`中获取`CANN` 包路径，如果不存在则默认为 `/usr/local/Ascend/ascend-toolkit/latest`                                                                                                                                                                     | 否  |
| -o，--output               | 输出文件路径，默认为当前路径                                                                                                                                                                                                                                                                    | 否  |
| -is，--input-shape         | 模型输入的shape信息，默认为空，例如"input_name1:1,224,224,3;input_name2:3,300",节点中间使用英文分号隔开。input_name必须是转换前的网络模型中的节点名称                                                                                                                                                                          | 否  |
| -d，--device               | 指定运行设备 [0,255]，可选参数，默认0                                                                                                                                                                                                                                                           | 否  |
| -dr，--dym-shape-range     | 动态Shape的阈值范围。如果设置该参数，那么将根据参数中所有的Shape列表进行依次推理和精度比对。(仅支持onnx模型)<br/>配置格式为："input_name1:1,3,200\~224,224-230;input_name2:1,300"。<br/>其中，input_name必须是转换前的网络模型中的节点名称；"\~"表示范围，a\~b\~c含义为[a: b :c]；"-"表示某一位的取值。 <br/>                                                                 | 否  |
| -ofs, --onnx-fusion-switch | onnxruntime算子融合开关，默认**开启**算子融合，如存在onnx dump数据中因算子融合导致缺失的，建议关闭此开关。使用方式：--onnx-fusion-switch False                                                                                                                                                                                  | 否  |
| --saved_model_signature | tensorflow2.6框架下saved_model模型加载时需要的签名。使用方式：--saved_model_signature serving_default，默认为serving_default                                                                                                                                                                             | 否  | |  |
| --saved_model_tag_set   | tensorflow2.6框架下saved_model模型加载为session时的标签，可根据标签加载模型的不同部分。使用方式：--saved_model_tag_set serve，默认为serve，目前支持传入多个tagSet，使用如：--saved_model_tag_set ['serve', 'general_parser']                                                                                                         | 否  | |  |
| --fusion-switch-file| 昇腾模型融合规则配置文件，传入该文件后，dump时会关闭文件中指定的融合规则，目前只支持在tensorflow2.6框架下使用。参数使用方法：--fusion-switch-file ./fusion_switch.cfg，其中fusion_switch.cfg文件配置方法参见：[关闭融合规则](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000105.html) | 否  |
| -dp, --device-pattern     | 设备模式，支持cpu和npu，目前npu模式下只支持saved_model模型。使用方式：-dp cpu                                                                                                                                                                                                                              | 否  | |  |
| --tf-json                 | 用于dump saved_model模型在cpu侧的算子集合json，当dump saved_model 模型在cpu的数据时，为必选参数                                                                                                                                                                                                             | 否  | |  |
| --exec | MindIE-Torch推理脚本的执行命令。使用方式：--exec "python bert_inference.py"                                                                                                                                                                                                                      | 否  | |  |
| -opname，--operation-name | MindIE-Torch场景下需要Dump的算子，默认为 all，表示会对模型中所有 op 进行 Dump，其中元素为MindIE-Torch算子类型，若设置 operation-name，只会 Dump 指定的 op。使用方式：--operation-name MatMulv2_1,trans_Cast_0                                                                                                                       | 否  | |  |
| -h    --help              | 用于查看全部的参数                                                                                                                                                                                                                                                                         | 否  | |  |

### 使用场景

| 使用示例                                                                                           | 使用场景                      |
|------------------------------------------------------------------------------------------------|---------------------------|
| [01_basic_usage](../../../examples/cli/debug/dump/01_basic_usage)                              | 基础示例，运行onnx模型dump         |
| [02_specify_input_data](../../../examples/cli/debug/dump/02_specify_input_data)             | 指定模型输入数据                  |
| [03_save_output_data](../../../examples/cli/debug/dump/03_save_output_data)                 | 指定结果输出目录                  |
| [04_specify_input_shape_info](../../../examples/cli/debug/dump/04_specify_input_shape_info) | 指定模型输入的shape信息(动态场景必须进行指定)。 |
| [05_caffe_model](../../../examples/cli/debug/dump/05_caffe_model)                           | 模型为Caffe框架的dump           |
| [06_saved_model](../../../examples/cli/debug/dump/06_saved_model)                           | 模型为tensorflow2.6框架下的 saved_model dump          |
| [07_mindie_torch_dump](../../../examples/cli/debug/dump/07_mindie_torch_dump)          | MindIE Torch场景-整网算子数据dump            |

### 查看Dump数据

读取、转换和保存 bin 数据的接口可以参考[API-读取和保存接口](../../llm/API-读取和保存接口.md)