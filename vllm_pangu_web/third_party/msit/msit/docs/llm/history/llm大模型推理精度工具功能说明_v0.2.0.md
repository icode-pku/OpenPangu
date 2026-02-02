## 新增特性

v0.2.0版本的新特性包括：

- 支持保存加速库layer拓扑信息（由msit llm dump --type layer开启）
- 支持保存模型cpu性能信息（由msit llm dump --type cpu_profiling开启）
- 支持保存加速库base算子信息（算子信息由msit llm dump --type op开启；kernel算子信息由msit llm dump --type kernel开启）

## Dump 特性

提供基本的加速库侧算子数据 dump 功能。

### 使用方式

```
msit llm dump --exec "bash run.sh patches/models/modeling_xxx.py"
```

### 参数说明

| 参数名                         | 描述                                                                                                                                                                                                                             | 必选 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| --exec                         | 指定拉起执行大模型推理脚本的命令，使用示例： --exec "bash run.sh patches/models/modeling_xxx.py"。**注：用户需自行保证执行命令的安全性，并承担因输入不当而导致的任何安全风险或损失；命令中不支持重定向字符，如果需要重定向输出，建议将执行命令写入 shell 脚本，然后启动 shell 脚本**                     | 是   |
| --type                         | dump类型，可选范围：['model', 'layer', 'op', 'kernel', 'tensor', 'cpu_profiling']，分别表示保存模型拓扑信息、layer拓扑信息、算子信息、kernel算子信息、tensor数据、profiling数据。默认为['tensor']。使用方式：--type layer tensor | 否   |
| -sd，--only-save-desc          | 只保存tensor描述信息开关，默认为否。使用方式：-sd                                                                                                                                                                                | 否   |
| -ids，--save-operation-ids     | 选择dump指定索引的tensor，默认为空，全量dump。使用方式：-ids 24_1,2_3_5                                                                                                                                                          | 否   |
| -er，--execute-range           | 指定dump的token轮次范围，区间左右全闭，可以支持多个区间序列，默认为第0次，使用方式：-er 1,3 或 -er 3,5,7,7（代表区间[3,5],[7,7],也就是第3，4，5，7次token。）                                                                    | 否   |
| -child，--save-operation-child | 选择是否dump所有子操作的tensor数据，仅使用ids场景下有效，默认为true。使用方式：-child True                                                                                                                                       | 否   |
| -time，--save-time             | 选择保存的时间节点，取值[0,1,2]，0代表保存执行前(before)，1代表保存执行后(after)，2代表前后都保存。默认保存after。使用方式：-time 0                                                                                              | 否   |
| -opname，--operation-name      | 指定需要dump的算子类型，支持模糊指定，如selfattention只需要填写self。使用方式：-opname self                                                                                                                                      | 否   |
| -tiling，--save-tiling         | 选择是否需要保存tiling数据，默认为false。使用方式：-tiling                                                                                                                                                                       | 否   |
| --save-tensor-part, -stp       | 指定保存tensor的部分，0为仅intensor，1为仅outtensor，2为全部保存，默认为2。使用示例：-stp 1                                                                                                                                      | 否   |
| -o, --output                   | 指定dump数据的输出目录，默认为'./'，使用示例：-o aasx/sss                                                                                                                                                                        | 否   |
| -h, --help            | 命令行参数帮助信息 | 否    |

### Dump 落盘位置

Dump默认落盘路径 `{DUMP_DIR}`在当前目录下，如果指定output目录，落盘路径则为指定的 `{OUTPUT_DIR}`。

- tensor信息会生成在默认落盘路径的ait_dump目录下，具体路径是 `{DUMP_DIR}/ait_dump/tensors/{device_id}_{PID}/{TID}`目录下。
- layer信息会生成在默认落盘路径的ait_dump目录下，具体路径是 `{DUMP_DIR}/ait_dump/layer/{PID}`目录下。
- cpu_profiling信息会生成在默认落盘路径的ait_dump目录下，具体路径是 `{DUMP_DIR}/ait_dump/cpu_profiling/{TIMESTAMP}/operation_statistic_{executeCount}.txt`。
- 算子信息会生成在默认落盘路径的ait_dump目录下，具体路径是 `{DUMP_DIR}/ait_dump/operation_io_tensors/{PID}/operation_tensors_{executeCount}.csv`。
- kernel算子信息会生成在默认落盘路径的ait_dump目录下，具体路径是 `{DUMP_DIR}/ait_dump/kernel_io_tensors/{PID}/kernel_tensors_{executeCount}.csv`。

注：`{device_id}`为设备号；`{PID}`为进程号；`{TID}`为 `token_id`；`{TIMESTAMP}`为时间戳；`{executeCount}`为 `operation`运行次数。

---

## Compare 特性

提供有精度问题的数据与标杆数据之间的比对能力。

### 使用方式

```
msit llm compare --golden-path golden_data.bin --my-path my-path.bin
```

#### 参数说明

| 参数名             | 描述                                       | 是否必选 |
| ------------------ | ------------------------------------------ | -------- |
| --golden-path, -gp | 标杆数据路径，支持单个数据文件路径或文件夹 | 是       |
| --my-path, -mp     | 待比较的数据路径，为单个数据文件路径       | 是       |
| --log-level, -l    | 日志级别，默认为info                       | 否       |
