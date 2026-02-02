# msit logits compare 工具
# 背景

通过logits dump工具获得bad case的部分token的Logits数据后，可以通过logits compare工具，对两个环境下dump的Logits数据进行比对。可以获得Logits之间的余弦相似度、KL散度、L1_Norm以及ULP，通过对上述参数进行分析，判断当前Logits比对是否通过，进而判断两种环境下bad case是否获得了非常接近的Logits结果，从而消除当前bad case在精度测评结果的影响。


# 介绍

logits compare工具主要功能是将两个环境下获取的Logits结果，按照文件名一致的对应规则，进行两两比对，并计算得出余弦相似度、KL散度、L1_Norm以及ULP四项指标结果。再进一步根据默认阈值或者用户输入的阈值，判断当前计算的指标结果是否达到预期。
满足以下任意一条标准，即可证明当前用例已达预期。
（1）余弦相似度、KL散度以及L1_Norm比对阈值通过。
（2）Logits中最大元素的误差值不超过数值精度的一个ULP（数值精度的最小单元）。

## 环境依赖

`torch`版本需使用`2.3.1`及以上版本，否则在调用`log_softmax`时，可能产生"无法应用于`float16`精度数据"的错误。

## 使用方法

```shell
msit llm logitscmp -gp {golden_logits_path} -mp {my_logits_path}  -cs {cosine_similarity} -kl {kl_divergence} -l1 {l1_norm} -d {dtype} -o {output_dir}
```

请将logits dump工具的结果输入当前工具中，golden-path和my-path要求是文件夹，且需要进行比对的Logits文件名称需要一一对应，如果存在未对应的文件，则工具提出警告，且该文件不参与比对。

## 参数说明

| 参数                 | 别名  | 描述                                                        | 是否必选 |
| -------------------- | ---- | ----------------------------------------------------------- | ------- |
| --golden-path        | -gp  | 存放标杆的Logits数据的文件夹                                  | 是      |
| --my-path            | -mp  | 存放待比对Logits数据的文件夹                                  | 是     |
| --cosine-similarity  | -cs  | 余弦相似度的比对阈值（默认0.999，取值范围：[-1, 1]）            | 否     |
| --kl-divergence      | -kl  | KL散度的比对阈值（默认0.0001，取值范围：[0, +∞)）              | 否     |
| --l1-norm            | -l1  | L1_Normd的比对阈值（默认0.01，取值范围：[-1, +∞)）             | 否     |
| --dtype              | -d   | 计算ULP时设定的数值精度（默认fp16，仅可输入bf16、fp16、fp32）   | 否     |
| --output-dir         | -o   | 结果输出文件夹（默认值"./output"）                             | 否      |
| --help               | -h   | 命令行帮助信息                                                | 否      |
| --log-level          | -l   | 设定日志级别（默认值"info"）                                   | 否      |

## 结果说明

使用logits compare工具会生成一个csv类型的结果文件，存储比对结果，以下是列名及含义：

| 列名              | 含义                                               |
| ----------------- | ------------------------------------------------- |
| file_name         | 比对的Logits文件名                                 |
| key               | bad case在整体数据集的下标                         |
| token_id          | 表示当前比对的是第几个token的Logits结果             |
| cosine_similarity | 余弦相似度的比对结果                                |
| kl_divergence     | KL散度的比对结果                                   |
| l1_norm           | L1_Norm的比对结果                                  |
| ulp_max_diff      | 两个Logits最大值之间的误差绝对值                    |
| ulp               | 最大Logits的数值精度的最小单元                      |
| passed            | 两个Logits比对是否通过 TRUE 表示通过，FALSE表示失败  |
| cmp_fail_reason   | 两个Logits无法进行比对的原因                        |

具体实例：

| file_name                   | key  | token_id | cosine_similarity | kl_divergence | l1_norm  | ulp_max_diff | ulp      | passed | cmp_fail_reason                         |
| --------------------------- | ---- | -------- | ----------------- | ------------- | -------- | ------------ | -------- | ------ |---------------------------------------- |
| logits_dev.jsonl_1352_0.pth | 1352 | 0        | 0.999998868       | 0.000956446   | 0.000365 | 0            | 0.015625 | TRUE   |                                         |
| invalid_file.pth            | NA   | NA       | NA                | NA            | NA       | NA           | NA       | NA     | parse_error:invalid_format_invalid_file |

针对无法比较的文件，所有指标列都会以`NA`结果输出，并在`cmp_fail_reason`列中说明无法比对的原因


# 示例

下列示例展示了如何通过命令行进行logits compare工作。

利用logits dump工具获得的Logits结果存储在对应的精度测评结果的data文件夹中，例如：`./output/data/NPU/precision_test/boolq/fp16/llama3_8b/logits/`，将两个环境下的这两个文件夹输入到工具中，工具将开始自动比对，并将结果输出到以`logits_cmp_res_`为前缀的csv结果文件中。

```shell
msit llm logitscmp -gp ./GPU/boolq/fp16/llama3_8b/logits -mp ./NPU/boolq/fp16/llama3_8b/logits
```
上述指令仅输入必须的Logits文件夹路径，比对阈值将按照默认值进行判断，将结果输入到默认`./output`文件夹下。

```shell
msit llm logitscmp -gp ./GPU/boolq/fp16/llama3_8b/logits -mp ./NPU/boolq/fp16/llama3_8b/logits -cs 0.9999 -kl 0.0001 -l1 0.001 -d fp32 -o result
```
上述指令输入了必须的Logits文件夹路径，以及各项指标的阈值，比对时将按照输入的阈值进行比对，且将结果输出到`./result/`文件夹下。

工具执行命令后获得存放在对应输出文件夹中，其中存放的文件名类似于：`logits_cmp_res_20250301151505.csv`，文件名以`logits_cmp_res`为前缀，拼接时间戳。
