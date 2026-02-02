# msit analyze功能使用指南

## 简介

模型支持度分析工具提供算子支持情况分析、算子定义是否符合约束条件和算子输入是否为空。


## 工具安装

- msit 工具安装请见 [msit一体化工具使用指南](../../docs/install/README.md)
- 安装好 msit 之后执行 msit install analyze 安装 analyze 组件


## 工具使用

一站式msit工具使用命令格式说明如下：

```shell
msit analyze [OPTIONS]
```

OPTIONS参数说明如下：

| 参数             | 说明                                                         | 是否必选 |
|----------------| ------------------------------------------------------------ | -------- |
| -gm, --golden-model | 标杆模型输入路径，支持onnx、caffe、tensorflow模型            | 是       |
| -o, --output   | 输出路径，在该路径下会生成分析结果**result.csv**             | 是       |
| --framework    | 模型类型，和[atc](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0005.html)参数一致，0：caffe，3：tensorflow，5：onnx | 否       |
| -w, --weight   | 权重文件，输入模型是caffe时，需要传入该文件                  | 否       |
| -soc, --soc-version | 芯片类型，不指定则会通过acl接口获取 | 否       |
| -h, --help | 命令行参数帮助信息 | 否       |


**特别说明**：当在Atlas200/500 A2推理产品上使用analyze工具进行模型支持度分析时，请手动指定-soc参数为具体的芯片类型。     

命令示例及输出如下：

```shell
msit analyze -gm /tmp/test.onnx -o /tmp/out
```

执行完成之后会在最后一行打印 analyze model finished。

输出结果在result.csv，会记录模型中每个算子的信息和支持情况，结果如下：

| ori_op_name           | ori_op_type        | op_name | op_type         | soc_type  | engine  | is_supported | details                                                      |
| --------------------- | ------------------ | ------- | --------------- | --------- | ------- | ------------ | ------------------------------------------------------------ |
| Reshape_46            | Reshape            |         | Reshape         | Ascend310 | AICORE  | TRUE         |                                                              |
| Cast_47               | Cast               |         | Cast            | Ascend310 | AICORE  | TRUE         |                                                              |
| Pad_49                | Pad                |         | PadV3           | Ascend310 | AICORE  | TRUE         |                                                              |
| Conv_52               | Convx              |         |                 | Ascend310 | UNKNOWN | FALSE        | No Op registered for Convx with domain_version of 11;Op is unsupported. |
| Transpose_53          | Transpose          |         | PartitionedCall | Ascend310 | AICORE  | TRUE         |                                                              |
| LeakyRelu_54          | LeakyRelu          |         | LeakyRelu       | Ascend310 | AICORE  | TRUE         |                                                              |
| BatchNormalization_60 | BatchNormalization |         | BatchNorm       | Ascend310 | AICORE  | TRUE         |                                                              |
| Shape_61              | Shape              |         | Shape           | Ascend310 | AICORE  | TRUE         |                                                              |

输出数据说明：

| 标题         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| ori_op_name  | 原始算子名称                                                 |
| ori_op_type  | 原始算子类型                                                 |
| op_name      | 模型迁移后算子名称                                           |
| op_type      | 模型迁移后算子类型                                           |
| soc_type     | 芯片类型                                                     |
| engine       | 算子执行引擎                                                 |
| is_supported | 算子是否支持，TRUE：支持，FALSE：不支持，可能原因包含算子不被当前硬件平台支持、算子定义不符合约束条件或算子输入为空，具体原因请参考details字段。 |
| details      | 算子支持情况问题描述，包括算子是否支持，算子定义是否符合约束条件、输入是否为空 |

## FAQ
- 使用过程中出现问题可先行查阅[FAQ](FAQ.md)