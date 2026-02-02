# msit logits dump 工具
# 背景

在大模型的精度测评工作中，不同设备之间产生的推理结果的精度会略有不同。对精度测评结果使用bad case分析工具后，可以获得同一模型同一数据集在不同环境下精度测评之后的`bad case`集合。针对这些错例集合，可以通过dump部分token的logits数据进一步比对分析，证明错例的真伪。因此当前logits dump工具提供`NPU`和`GPU`环境下对模型推理的logits结果落盘的功能。


# 介绍

logits dump工具主要功能是对模型推理过程中的Logits进行落盘。首先分析bad case结果，在内存中将该结果与全量数据集取交集，在推理过程中仅推理这些错例。在错例集推理过程中，保存每个错例case的固定长度token的Logits结果。

**说明** ：模型推理过程依赖modeltest工具，请在使用logits dump工具前确保modeltest可用。

## 参数说明

由于工具调用了modeltest工具，所以需要传入modeltest精度测评的指令，且**batchsize参数需设定为1**。

| 参数                  | 别名   | 描述                                                                          | 是否必选 |
| --------------------- | ----- | ----------------------------------------------------------------------------- | ------- |
| --exec                | -e    | modeltest精度测评指令，需设定batchsize为1（命令需以'torchrun'或'modeltest'开头）。**注：用户需自行保证测评命令的安全性，并承担因输入不当而导致的任何安全风险或损失**  | 是     |
| --bad-case-result     | -bcr  | bad case分析工具的结果文件，存储着两种场景下的精度测评结果的错例集                  | 是     |
| --token-range         | -tr   | 需要dump的前多少条token的logits（默认值为1）                                     | 否     |
| --help                | -h    | 命令行帮助信息                                                                  | 否     |
| --log-level           | -l    | 日志等级（默认值为info）                                                        | 否     |


## 使用方法

```shell
msit llm logitsdump -e {modeltest_cmd} -bcr {bad_case_csv_path} [ -tr {int} ]
```

传入modeltest的精度测评命令需要设定batchsize为1，以便dump下来的数据可以直接在`Logits Compare`工具中正常比对。可选参数可以配置dump数据的范围。

**说明**：
  - 模型推理过程中请按照modeltest工具的数据集准备方法，将数据集配置到当前指令运行的文件夹下，或者在task_config文件中设定本地数据集地址参数。
  - humanevalx数据集是代码评测类数据集，推理时额外需要modeltest的tools文件夹，需要手动将MindIE-LLM仓库下的examples/atb_models/tests/modeltest/tools/文件夹，复制到当前运行指令的文件夹下，或者在examples/atb_models/tests/modeltest/路径下运行当前工具。
  - 当设定的`--token-range`超过output token的个数时，仅会dump所有token个数的Logits。

# 示例
下列示例展示了如何通过命令行进行logits dump工作。

利用bad case分析工具获得BoolQ数据集在不同环境下的bad case结果文件，例如：`msit_bad_case_result_20250117134959.csv`

```shell
msit llm logitsdump -e "modeltest --model_config_path modeltest/config/model/llama3_8b.yaml --task_config_path modeltest/config/task/boolq.yaml --batch_size 1 --tp 1 --output_dir ./outputs --lcoc_disable --save_debug_enable" -bcr ./msit_bad_case_result_20250117134959.csv -tr 10
```

上述命令行传入了单卡拉起modeltest在BoolQ数据集上单batch的精度测评命令，以及bad case结果地址，同时设定了落盘前10个token的Logits数据。多卡指令需要使用torchrun拉起modeltest，详见modeltest的资料。

最终会获得存放在类似`./outputs/data/{device}/precision_test/boolq/{data_type}/llama3-8b/logits/`文件夹路径下的所有Logits结果。其中存放的文件名类似于：`logits_{task_name}_{batch_id}_{token_id}.pth`，每个文件中保存了一个batch的单个token的Logits结果。
