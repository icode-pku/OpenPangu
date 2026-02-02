# msserviceprofiler 工具介绍

## 概述

msserviceprofiler 是一款基于昇腾平台，支持MindIE Service框架和vLLM框架的服务化调优工具。

其性能采集与数据解析能力已嵌入昇腾CANN工具包，支持MindStudio Insight、Chrome Tracing、Grafana多个平台数据可视化。

目前，扩展能力**服务化性能数据比对工具**和**vLLM服务化性能采集工具**已在本仓库开源。

## 硬件环境
|   类型    | 配置参考 |  服务化性能数据比对工具 | vLLM服务化性能采集工具 |
|:-------:|:-------:|:-------:|:-------:|
|   服务器   | Atlas 800I A2 推理产品   | √ | √ |
|   推理卡   | Atlas 300I Duo 推理卡+Atlas 800 推理服务器（型号：3000）   | √ |  |

## 特性清单

### [服务化性能数据比对工具](docs/服务化性能数据比对工具.md)

支持对使用msserviceprofiler工具采集的性能数据进行差异比对，通过比对快速识别可能存在的问题点。

### ️[vLLM服务化性能采集工具](docs/vLLM服务化性能采集工具.md)

基于Ascend-vLLM，提供性能数据采集能力，结合msserviceprofiler的数据解析与可视化能力，可以vLLM服务化推理调试调优。

### ️[服务化自动寻优工具](docs/serviceparam_optimizer_instruct.md)

基于msserviceprofiler工具采集的性能数据，提供服务化参数自动寻优能力，可以对服务化的参数以及测试工具的参数进行寻优。

### ️[服务化专家建议工具](docs/service_profiling_advisor_instruct.md)

基于benchmark 输出结果以及 service 的 config.json 配置，提供分析提高 TFTT / Throughput 等的优化点能力。

### ️[服务化多维度解析工具](docs/服务化多维度解析工具.md)

基于msserviceprofiler工具采集的性能数据，提供性能数据多维度分析能力，可以对性能数据进行batch维度、request维度和service维度分析。

### ️[服务化拆解工具](docs/service_performance_split_instruct.md)

基于msserviceprofiler工具采集的性能数据，提供性能数据拆解能力，可以对batch内各阶段耗时进行分析。


## 安装指南

目前工具支持pip装包后调用和源码下载，脚本调用两种方式。
#### pip 安装 msserviceprofiler
```shell
pip install -U msserviceprofiler
```
#### 或通过源码方式使用 msserviceprofiler

```shell
git clone https://gitcode.com/Ascend/msit.git
export PYTHONPATH=$PWD/msit/msserviceprofiler/:$PYTHONPATH
```

## 支持与帮助

🐛 [Issue提交](https://gitcode.com/Ascend/msit/issues)

💬 [昇腾论坛](https://www.hiascend.com/forum/forum-0106101385921175006-1.html)
