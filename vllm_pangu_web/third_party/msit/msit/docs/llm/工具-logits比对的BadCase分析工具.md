# msit bad case 分析工具
# 背景 
在大模型推理精度定位场景下，通常会出现两个模型在同一数据集下，表现不一致的情况。比如，昇腾模型在 `NPU` 上，经过数据集评测，发现有若干问题出现回答错误的情况，但是这些同样的问题，其原生模型在 `GPU` 上的结果却是正确的，这时我们就称这两个问题为 `bad case`。

`msit` 工具针对这种场景，提供"bad case分析工具"进行自动分析，使能用户快速定位。后续衔接"logits dump工具"落盘bad case在模型推理时的logits数据，使用"logits 比对工具"进行对比logits，实现进一步的推理精度测评以及精度问题定位。


# 介绍
bad case分析工具的主要功能是对输入的两个场景下的数据集精度测评结果进行分析，并输出对应 `bad case` 。比对同一条query的 `NPU` 和 `GPU` 环境下的回答结果，当在 `NPU` 上的回答结果与 `GPU` 上的回答结果不一致时，那么这个问题本身将会被当作为 `bad case` 进行输出。

  **说明** ：精度测评结果是对数据集每条query运行结果的统计。

## 分析逻辑

一般而言，精度测评结果中有以下几列：

| 列名             | 描述                                                      | 是否必须存在 |
| ---------------- | -------------------------------------------------------- | ------ |
| key              | 数据集中每条query的标识                                    | 是     |
| queries          | 数据集中每条query的内容                                    | 否     |
| input_token_ids  | 数据集中每条query的token id                                | 否     |
| output_token_ids | 数据集中每条query推理完成后的结果的token id                 | 否     |
| test_result      | 模型推理的结果                                            | 否     |
| golden_result    | 数据集每条query的正确结果                                  | 否     |
| pass             | 比对模型推理结果和正确结果，判断模型推理在当前query是否通过   | 是     |

根据`key`和`queries`列，结合两个精度测评表格，`GPU`精度测评表格列名加后缀`_golden`，`NPU`精度测评表格列名加后缀`_test`。比对表格中的两个`pass`列，筛选出后缀为`_golden`列与后缀为`_test`列结果不同的行，保存下来作为bad_case结果。

## 使用方法

```shell
msit llm bcanalyze -gp {golden_path} -mp {my_path}  [可选参数] 
```

用户可以传入数据集精度测评结果的 `csv` 路径，可以一键式进行 `bad case` 分析

参数介绍如下：

| 参数名           | 别名   | 描述                               | 是否必选 |
| --------------- |--------| ---------------------------------- | ------- |
| --golden-path   | -gp    | 标杆环境数据集精度测评结果文件地址    | 是      |
| --my-path       | -mp    | 测试环境数据集精度测评结果文件地址    | 是      |
| --help          | -h     | 命令行帮助信息                      | 否      |
| --log-level     | -l     | 日志等级（默认值为info，可选值：debug, info, warning, error, fatal, critical）  | 否     |


# 示例
下列示例展示了如何通过命令行进行 `bad case` 分析

通过BoolQ数据集进行精度测评，获得数据集精度测评结果，例如：

```sh
BoolQ_fa_batch16_tp2_full_2025_01_17_15_34_22_debug_info.csv # GPU环境下对BoolQ数据集精度测评的结果
BoolQ_pa_batch16_tp2_full_2025_01_17_15_25_20_debug_info.csv # NPU环境下对BoolQ数据集精度测评的结果
```

在命令行中使用：
```sh
msit llm bcanalyze -gp "BoolQ_fa_batch16_tp2_full_2025_01_17_15_34_22_debug_info.csv" -mp "BoolQ_pa_batch16_tp2_full_2025_01_17_15_25_20_debug_info.csv"
```

获得存放在类似`msit_bad_case/analyzer/msit_bad_case_result_20250117134959.csv`路径下的`bad case`分析结果。结果中必定包含`key`列和两个`passed`列数据，以及可能包含其他列数据（例如input_token_ids、output_token_ids等），通过展示这些数据列表明bad case的结果。
