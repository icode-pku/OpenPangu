# TFDebugRunner使用说明

## 简介
- 此脚本用于dump TensorFlow1.x框架下.pb模型在GPU上的算子输出数据，基于TensorFlow Debugging工具（tf_debug）。
- 此功能包含在msit debug dump中，直接运行该脚本较复杂，不建议直接使用

## 工具安装
- 安装命令：
- **注:** 目前 dump 相关的功能安装集成到了 debug compare 命令里，用户只需要安装debug compare即可实现debug dump功能，命令如下：
```shell
msit install compare
```
## 使用方法

### 功能介绍

#### 安全风险提示

在模型文件传给工具加载前，用户需要确保传入文件是安全可信的，若模型文件来源官方有提供SHA256等校验值，用户必须要进行校验，以确保模型文件没有被篡改。

#### 使用入口
在**保存数据的路径**下运行tf debug runner相关命令，命令示例如下，**其中路径需使用绝对路径**
  ```sh
  python3 /msit_path/msit/components/debug/compare/msquickcmp/tf_debug_runner.py -m /model_path/tf/model.pb -i /path/tf/input.bin -o /output_path/tf/output --output_nodes "output_node:0"
  ```
视图进入调试命令交互模式tfdbg，执行**run**命令
```commandline
tfdbg> run
```
**run**命令执行完成后，参考CANN资料[准备GPU侧npy数据文件](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/modelaccuracy/atlasaccuracy_16_0006.html)
**收集npy数据文件**章节进行数据收集
#### 输出结果说明

```sh
{output_path}  # 设定的保存文件路径
├-- {op_name}.{output_index}.{timestamp}.npy #dump的npy数据文件
```
#### 命令行入参说明

| 参数名              | 描述                                                                                                                                                                                                      | 必选 |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----|
| -m，--model-path  | 模型文件 [.pb]路径，其中saved_model.pb为TF2.6.5版本模型文件                                                                                                                                                             | 是  |
| -i，--input-path  | 模型的输入数据路径，多个输入以逗号分隔，例如：/home/input\_0.bin,/home/input\_1.bin,/home/input\_2.npy。注意：使用aipp模型时该输入为om模型的输入,且支持自动将npy文件转为bin文件                                                             | 是  |
| -o，--out-path    | 输出文件路径，默认为当前路径                                                                                                                                                                                          | 否  |
| -s，--input-shape | 模型输入的shape信息，默认为空，例如"input_name1:1,224,224,3;input_name2:3,300",节点中间使用英文分号隔开。input_name必须是转换前的网络模型中的节点名称                                                                                                | 否  |
| --output-nodes   | 用户设定的需要输出的算子                                                                                                                                                                                            | 是  |
| -h    --help     | 用于查看全部的参数                                                                                                                                                                                               | 否  | |  |