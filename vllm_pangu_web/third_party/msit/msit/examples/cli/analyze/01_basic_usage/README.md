# Basic Usage

## 介绍

Analyze分析工具提供从Caffe、TensorFlow以及ONNX模型迁移至昇腾硬件平台上的模型支持度分析功能。工具涵盖**ATC支持度分析**、**昇腾算子速查工具分析**以及**ONNXChecker算子约束分析**三大分析手段，对算子支持情况、算子定义是否符合约束条件以及算子输入是否为空等主要场景进行判断，并给出详细分析报告。

## 运行原理
1. Analyze工具首先调用ATC转换工具将给定待评估模型转换为包含模型所有算子信息的json格式；
2. 根据ATC转换工具返回的转换结果进行不同分析：
    - EVAL_ATC_SUCCESS：ATC模型转换成功，而后根据om模型信息更新所有融合算子与非融合算子的支持度信息
    - EVAL_ATC_UNSUPPORTED_OP_ERR：ATC转换结果显示，模型中存在不支持的算子。先更新不支持算子信息，再使用昇腾算子速查工具分析当前昇腾设备上算子的支持情况。若传入的模型为ONNX格式，则将调用ONNXChecker类对模型中算子约束条件进行检查
    - EVAL_ATC_OTHER_ERR：其他不支持的类型，将对模型使用昇腾算子速查工具算子支持分析，若为ONNX格式，则继续进行算子约束条件检查
3. 工具运行结果将保存至用户指定的csv文件中，详细解读见`使用示例`章节

## 使用示例

```shell
msit analyze [OPTIONS]
```

OPTIONS参数说明如下：

| 参数             | 说明                                                         | 是否必选 |
|----------------| ------------------------------------------------------------ | -------- |
| -gm, --golden-model | 标杆模型输入路径，支持onnx、caffe、tensorflow模型            | 是       |
| -o, --output   | 输出路径，在该路径下会生成分析结果**result.csv**             | 是       |
| --framework    | 模型类型，和[atc](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/atctool/atlasatcparam_16_0014.html)参数一致，0：caffe，3：tensorflow，5：onnx | 否       |
| -w, --weight   | 权重文件，输入模型是caffe时，需要传入该文件                  | 否       |
| -soc, --soc-version | 芯片类型，不指定则会通过[acl](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclpythondevg_01_0009.html)接口获取 | 否       |

**特别说明**：当在Atlas200/500A2推理产品上使用analyze工具进行模型支持度分析时，请手动指定-soc参数为具体的芯片类型。

## 运行示例

```shell
msit analyze -gm /tmp/test.onnx -o /tmp/out
```

```shell
2023-05-11 11:23:25,824 INFO : convert model to json, please wait...
2023-05-11 11:23:28,210 INFO : convert model to json finished.
2023-05-11 11:23:29,997 INFO : try to convert model to om, please wait...
2023-05-11 11:23:35,127 INFO : try to convert model to om finished.
2023-05-11 11:23:36,321 INFO : analysis result has bean written in /tmp/result.csv
2023-05-11 11:23:36,321 INFO : analyze model finished.
```

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

| 标题         | 说明                                                             |
| ------------ |----------------------------------------------------------------|
| ori_op_name  | 原始算子名称                                                         |
| ori_op_type  | 原始算子类型                                                         |
| op_name      | 模型迁移后算子名称                                                      |
| op_type      | 模型迁移后算子类型                                                      |
| soc_type     | 芯片类型                                                           |
| engine       | 算子执行引擎                                                         |
| is_supported | 算子是否支持，TRUE：支持，FALSE：不支持，可能原因包含算子不被当前硬件平台支持、算子定义不符合约束条件或算子输入为空 |
| details      | 算子支持情况问题描述，包括算子是否支持，算子定义是否符合约束条件、输入是否为空                        |

