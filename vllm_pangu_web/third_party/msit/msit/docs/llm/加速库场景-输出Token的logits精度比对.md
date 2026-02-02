# 输出 token 的 logits 精度比对 - 加速库推理场景

在加速库推理场景，当存在精度问题时，用户需要排查到底是哪个输出 token 出现了精度误差。以下介绍如何对输出 token 进行精度比对。

## 对比思路

使用方法分为三步：

1. dump torch 模型 token 的输出 logits
2. dump 加速库 token 的输出 logits
3. 将上面两个步骤生成的 dump 文件，输入到 msit llm compare 中完成比对

> 请注意，在 dump 时，请将 topK 配置为 top1，可以确保推理结果相同情况下，torch 和 dump 选择的都是同一个 token

## 第一步：dump torch 模型 token 的输出 logits

使用[PyTorch 场景的精度数据采集](/msit/docs/llm/工具-Pytorch场景数据dump.md)。将模型最后的输出的 logits dump 下来。
```python
from msit_llm import DumpConfig, register_hook #在模型py文件中文件开头导入DumpConfig和register_hook

# dump_last_logits=True 参数表示要 dump 输出 logits
# dump_path="./torch_dump" 参数指定 dump 保存路径
dump_config = DumpConfig(dump_last_logits=True, token_range=list(range(1000)), dump_path="./torch_dump")
register_hook(model, dump_config)  # model是要dump中间tensor的模型实例，在模型初始化后添加代码

```

> dump 数据路径：
> - 参数`dump_last_logits=True`表示要 dump 输出 logits
> - 参数`token_range`指定要 dump 的 token 范围。数据类型为list，默认为`[0]`，即只dump第0个token的数据。`token_range=list(range(1000))`表示要 dump 0-999 个 token 的数据，可以根据实际情况自行调整
> - 参数`dump_path`指定 dump 保存路径，默认为当前路径。可以根据实际情况自行决定
> - logits 信息落盘位置：在`{DUMP_DIR}/msit_dump_{TIMESTAMP}/torch_tensors/{cuda|cpu|npu}{device_id}_{PID}/{TID}`目录下，会输出所有 token 的 logits
>   注：`device_id`为设备号；`PID`为进程号；`TID`为`token_id`

## 第二步：dump 加速库 token 的输出 logits

使用[加速库 Dump 功能](./工具-DUMP加速库数据使用说明.md)。需要 Dump 两次：

1. 先 dump 网络模型，获取到网络最后一层的编号
  * 如果使用mindie RC1 及之后版本，atb模型都存在 prefill 和 decode 两个模型，所以本来32层的会变成64层。获取最后一层编号比如63，还需要加上31
2. 再通过编号，指定 Dump 最后一层的输出

### 使用方式

1. 先 dump 网络模型，获取到网络最后一层的编号

```bash
  msit llm dump --exec "bash run.sh" --type layer
```

> - `--exec`参数用于指定拉起执行大模型推理脚本的命令。**用户需自行保证命令的安全性，并承担因输入不当而导致的任何安全风险或损失**
> - `--type layer`参数是指定 dump 模型的 layer 层信息
> - layer 信息落盘位置：在`{DUMP_DIR}/msit_dump_{TIMESTAMP}/layer/{PID}/`目录下，可以通过 layer 信息，获取到最大的编号，编号最大的即是模型最后一层

2. 通过编号，指定 Dump 最后一层的输出

```bash
  msit llm dump --exec "bash run.sh" --type model tensor -ids 上一步骤获取的编号 -er 0,1000 -child False -stp 1
```

> - `--exec`参数用于指定拉起执行大模型推理脚本的命令。**用户需自行保证命令的安全性，并承担因输入不当而导致的任何安全风险或损失**
> - `--type model tensor`参数是指定 dump 模型的 model 拓扑信息，以及最后一层输出 tensor
> - `-ids 上一步骤获取的编号`参数是指定 dump 的最后一层。例如：31,63
> - `-er 0,1000`参数指定 dump 的 token 范围，可以根据实际情况自行决定
> - `-child False`参数指定不 dump 子操作的 tensor，减少所需空间
> - `-stp 1`参数指定只 dump 算子的输出
> - tensor 信息落盘位置：tensor 信息落盘位置：在`{DUMP_DIR}/msit_dump_{TIMESTAMP}/tensors/{device_id}_{pid}/{TID}`目录下

## 第三步：使用 msit llm compare 完成比对

使用[自动比对功能](/msit/docs/llm/工具-大模型精度比对.md)。比对标杆数据和加速库数据。

### 使用方式

```shell
msit llm compare -gp {DUMP_DIR}/msit_dump_{TIMESTAMP}/torch_tensors/{cuda|cpu|npu}{device_id}_{PID}/ -mp msit_dump_{TIMESTAMP}/tensors/{device_id}_{PID}/ -cl logits
```

msit llm compare 提供有精度问题的数据与标杆数据之间的比对能力。

> - --golden-path 参数为第一步中 `torch_tensor`所在目录 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/torch_tensors/{cuda|cpu|npu}{device_id}_{PID}/`
> - --my-path 参数为第二步中 `atb_tensor`所在目录 `{DUMP_DIR}/msit_dump_{TIMESTAMP}/tensors/{device_id}_{PID}/`
> - -cl logits 表示logits比对
> - 完成比对后会在 `output_dir`下生成一个 `msit_cmp_report_{TIMESTAMP}.csv`，保存比对的最终结果。
> - csv 报告查看参考[精度比对结果参数说明](/msit/docs/llm/精度比对结果参数说明.md)
