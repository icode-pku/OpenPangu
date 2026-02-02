# 单算子精度预检

## 介绍
该工具能够对那些未被编译优化的算子进行精度预检，可以用于快速定界误差累计问题，检测kernel级别的算子精度是否达标。

## 环境依赖

```sh
pip install torch==1.11.0
```

## 运行示例
### 输入数据
执行以下命令，dump模型推理过程的tensor数据。
`msit debug dump -m /home/HwHiAiUser/prouce_data/resnet_offical_saved_model -dp npu -is {input_shape} -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test`，其中`{input_shape}`是模型的输入shape，需要根据实际情况修改。
- 相关参数使用参考[saved_model dump功能使用参考](../../dump/06_saved_model/README.md)
- 生成的tensor信息会落盘在-o参数指定的目录下

### 使用入口
预检功能可以直接通过msit命令行工具启动精度对比。启动方式如下：
命令示例中**路径需使用绝对路径**，**-i参数指定的tensor路径是由msit debug dump 命令落盘的**
```sh
msit debug opcheck -i /home/HwHiAiUser/result/test/{timestamp} -o /home/HwHiAiUser/result/test/opcheck_test
```

## 输出结果说明
输出result_{timestamp}.xlsx的预检结果文件，包含2个sheet页，分别为opcheck_results（预检成功的算子）和addition_failed_cases（预检失败的算子）。
### 预检结果列名说明
| 列名                    | 说明                                     |
|-----------------------|----------------------------------------|
| op_type               | 算子类型，格式为以list包含的算子类名                   |
| op_name               | 算子名称，以'_'分隔的算子拓扑结构名                    |
| op_param              | 算子参数                                   |
| tensor_path           | 算子输入intensor及输出outtentor的文件名称          |
| out_tensor_id         | 算子输出outtensor的序号（部分算子输出可能有多个outtensor） |
| rel_precision_rate(%) | 实际的精度通过率（使用相对误差，全部通过则为100%）            |
| max_rel_error         | 最大的相对误差值                               |
| abs_precision_rate(%) | 实际的绝对误差精度通过率                           |
| max_abs_error         | 最大的绝对误差值                               |
| cosine_similarity     | 余弦相似度                                  |
| kl_divergence         | kl散度                                   |
| fail_reason           | 失败原因，包括精度未达到给定的标准或算法执行失败原因             |

相关计算公式为：
```
相对误差：rel_error = abs(actual_output - golden_output) / abs(golden_output)
精度通过率：rel_precision_rate = sum(rel_error <= etol) / size(rel_error) * 100
精度比对结果：precision_result = bool(rel_precision_rate >= pass_rate)
```
## 目前精度预检算子支持情况
| 算子名称(A-G)     | 算子名称(H-M)  | 算子名称(N-R)  | 算子名称(S-Z)   |
|---------------|------------|------------|-------------|
| Add           | LogicalOr  | Pad        | SoftMaxV2   |
| Adds          | LogicalAnd | PadD       | Sub         |
| BatchNorm     | Less       | Pack       | Sigmoid     |
| BatchMatMulV2 | Mul        | ReduceMean | StrideSlice |
| BiasAdd       | Minimum    | Rsqrt      | Tanh        |
| BNInfer       | Mul        | ReduceMean | Tile        |
| ConcatV2      | MatMulV2   | ReduceSum  | Transpose   |
| Conv2D        |            | Relu       |             |
| ClipByValue   |            |            |             |
| GatherV2      |            |            |             |

## FAQ
1. 为什么需要使用PyTorch 1.11.0版本？

标杆实现采用PyTorch 1.11.0版本，如果使用其他版本的PyTorch，可能导致算子实现不同。

2. 为什么BatchNorm算子预检会出现异常？

BatchNorm算子的标杆实现采用了PyTorch的torch.ops.aten.native_batch_norm实现，该实现在计算过程中可能会出现inf/nan等异常场景。