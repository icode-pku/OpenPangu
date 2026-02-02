# 融合算子匹配对应pass

## 1 简介

融合算子匹配对应pass功能主要用于提供融合算子对应的pass相关信息，快速定位有问题的精度pass。

## 2 使用方式

### 前置准备
1. 请使用[安装](../../install/README.md)中的源码安装msit及其compare组件。

2. 使用msit debug dump功能

    分别进行以下两种情况的 dump：

    - 正常推理：直接运行推理任务并生成 dump 数据。
    - 关闭融合：使用 --fusion-switch-file 参数传递关闭融合规则的文件，执行推理任务并生成 dump 数据。关闭融合规则文件的格式和内容可以参考[关闭融合规则文件说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/developmentguide/appdevg/aclcppdevg/aclcppdevg_000105.html)。

    具体使用参考[dump使用方法](../dump/README.md)

### 精度比对获得关联pass

1. 假设源码被克隆到`/home/HwUser/fusion_pass`。
2. 调用以下脚本查看help信息
    ``` bash
    python /home/HwUser/fusion_pass/msit/msit/components/debug/compare/fusion_pass_cmp/get_fusion_pass.py -h
    ```
    | 参数 | 是否必选 | 介绍 |
    | --- | --- | --- |
    |-m, --fusion-after-dir | 是 | str，开启融合的精度数据目录 |
    |-g, --fusion-before-dir | 是 | str，关闭融合的精度数据目录 |
    |-f, --fusion-after-model | 是 | str，开启融合的模型json文件 |
    |-cf, --fusion-before-model | 是 | str，关闭融合的模型json文件 |
    |-o, --output-path | 否 | str，输出目录，默认是当前目录 |
    |-fn, --fusion-node-switch | 否 | 开关参数，是否只输出融合算子的比对结果，默认为True |


## 3 输出结果说明

- **比对结果**：在指定的输出文件 `fusion_pass_info.csv` 中，输出正常推理情况下算子的output数据与关闭融合output数据的比对情况，并给出融合算子对应的PassName。
* 比对结果示例及说明如下：

| 关注项 / 结果项 | OpType | NPUDump                             | DataType | Address | GroundTruth  | TensorIndex     | Shape          | CosineSimilarity                              | ... | MeanRelativeError                             | PassName                                         | MatchError  |
|-----------|--------|-------------------------------------|----------|---------|--------------|-----------------|----------------|-----------------------------------------------|-----|-----------------------------------------------|--------------------------------------------------|-------------|
| 示例值       | Conv2D | Conv2D                              | float32  | NaN     | Conv2D,Pad   | Conv2D:output:0 | [1,64,112,112] | 1                                             | ... | 0.000364                                      | "{'PadConv2dFusionPass', 'AABiasaddConvFusion'}" |             |
| 说明        | 算子类型   | 正常推理模型中的算子，由于融合规则，可能会对应多个关闭融合情况下的算子 | 数据类型     | -       | 关闭融合情况下的模型算子 | -               | -              | 各类误差比对类型结果，主要需要看是否某一项超过精度阈值(即某项异常)，若超过则需要重点关注 | -   | 各类误差比对类型结果，主要需要看是否某一项超过精度阈值(即某项异常)，若超过则需要重点关注 | 融合算子对应的融合规则                                      | 标识未匹配上的融合算子 |

 具体阈值可参考 [精度比对结果：比对结果分析](../../../examples/cli/debug/compare/result_analyse/README.md#比对结果分析)
