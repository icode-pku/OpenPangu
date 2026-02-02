# msit debug opcheck功能使用指南
## 简介
opcheck（精度预检）提供了传统小模型场景下kernel级别的精度预检功能，支持对经过GE推理后dump落盘数据进行算子精度预检，检测kernel级别的算子精度是否达标。

**注:** 目前只支持TensorFlow2.6.5，未来会支持更多框架

## 环境准备
### 1. TensorFlow2.6.5
```
pip3 install tensorflow==2.6.5 --no-index --find-links  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html --trusted-host ascend-repo.obs.cn-east-2.myhuaweicloud.com
```
### 2.PyTorch
```
pip install torch==1.11.0
```
### 3.工具安装
- 首先安装msit，安装教程请见 [msit一体化工具使用指南](../../install/README.md)
- 然后安装opcheck组件，执行命令：`msit install opcheck`

## 使用方法

### 参数说明
| 参数名                      | 描述                                                         | 是否必选 |
| --------------------------- | ------------------------------------------------------------ | -------- |
| --input, -i                 | tensor数据路径，为文件夹，由msit debug dump 落盘，示例：{output_path}/{timestamp} | 是       |
| --output, -o                | 输出文件的保存路径，默认为当前路径                | 否       |
| --mode, -m                |    指定预检模式，可选：'single'、'autofuse'，默认为`single`       | 否       |
| --graph-path, -gp                |     指定GE dump的模型图定义文件     | 否       |
| --log-level, -l                |       设置日志等级，默认为`info`      | 否       |
| --help, -h              | 命令行参数帮助信息| 否       |

### 使用场景

|                    使用示例                                               | 使用场景                      |
|---------------------------------------------------------------------------|---------------------------|
| [单算子精度预检](../../../examples/cli/debug/opcheck/01_single_opcheck/README.md)                | 支持对未参与编译优化的单算子进行精度预检   |
| [自动融合算子精度预检](../../../examples/cli/debug/opcheck/02_autofuse_opcheck/README.md)            | 支持对自动融合生成的融合算子进行精度预检     |
