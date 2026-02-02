# msit convert功能使用指南

## 简介
convert模型转换工具依托ATC（Ascend Tensor Compiler），AOE（Ascend Optimization Engine），提供由ONNX、TensorFlow、Caffe、MindSpore模型至om模型的转换及调优功能。

* ATC (Ascend Tensor Compiler)
> 昇腾张量编译器（Ascend Tensor Compiler，简称ATC）是异构计算架构CANN体系下的模型转换工具， 它可以将开源框架的网络模型以及Ascend IR定义的单算子描述文件（json格式）转换为昇腾AI处理器支持的.om格式离线模型。
>
> 模型转换过程中，ATC会进行算子调度优化、权重数据重排、内存使用优化等具体操作，对原始的深度学习模型进行进一步的调优，从而满足部署场景下的高性能需求，使其能够高效执行在昇腾AI处理器上。
>
> [更多说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0005.html)
* AOE (Ascend Optimization Engine)
> AOE（Ascend Optimization Engine）是一款自动调优工具，作用是充分利用有限的硬件资源，以满足算子和整网的性能要求。
>
> AOE通过生成调优策略、编译、在运行环境上验证的闭环反馈机制，不断迭代出更优的调优策略，最终得到最佳的调优策略，从而可以更充分利用硬件资源，不断提升网络的性能，达到最优的效果。
>
> [更多说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/auxiliarydevtool_0014.html)


## 工具安装

- 工具安装请见 [msit一体化工具使用指南](../install/README.md)

## 工具使用

一站式msit工具使用命令格式说明如下：
```shell
msit convert [subcommand]
```
msit convert目前支持以下3种子命令：

| subcommand | 说明                      |
| ---------- | ------------------------- |
| atc        | 使用atc进行模型转换       |
| aoe        | 使用aoe进行模型转换及调优 |

### atc命令
使用ATC后端进行模型转换，命令格式如下：
```shell
msit convert atc [args]
```
参数定义严格遵从ATC的参数定义，由于参数较多，详情可参考：https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasatc_16_0039.html#ZH-CN_TOPIC_0000001949484154__section6351244132417
使用示例：
```shell
msit convert atc --model resnet50.onnx --framework 5 --soc_version <soc_version> --output resnet50
```
### aoe命令
使用AOE后端进行模型转换，命令格式如下：
```shell
msit convert aoe [args]
```
参数定义严格遵从AOE的参数定义，由于参数较多，详情可参考：https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/auxiliarydevtool_0014.html

使用示例：
```shell
msit convert aoe --model resnet50.onnx --job_type 2 --output resnet50
```

## FAQ
使用convert组件进行模型转换时如遇问题，请先行查阅[FAQ](FAQ.md)