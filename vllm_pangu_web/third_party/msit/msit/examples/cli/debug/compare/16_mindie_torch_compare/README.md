
# 基于MindIE-Torch 精度比对场景

## 1. 相关依赖

- CANN（8.0RC3及以上）
- MindIE（1.0RC3及以上）

## 2. msit安装

- msit安装请参考[一体化安装指导](/msit/docs/install/README.md)

## 3. Dump 数据 

- MindIE-Torch 数据Dump请参考[MindIE-Torch场景-整网算子数据dump](../../dump/07_mindie_torch_dump/README.md)

## 4. Compare 精度对比 

 - 执行 `msit debug compare --golden-path [/path/to/cpu/dumpdata] --my-path [/path/to/npu/dumpdata] --output [path/to/csv] --ops-json [path/to/json]`，输出比对结果 csv 文件

### 参数说明

| 参数名                 | 描述                                                         | 必选 |
| ---------------------- | ------------------------------------------------------------ | -------- |
| --golden-path, -gp     | CPU/GPU Dump数据根路径                   | 是       |
| --my-path, -mp         | NPU Dump数据根路径                         | 是       |
| --ops-json | 运行 `msit debug dump` 时产生的算子映射关系文件路径，通常在当前文件夹下 | 是       |
| --output, -o           | 比对结果csv的输出路径，默认为当前路径              | 否       |

## 注意

- MindIE场景算子精度对比目前支持TorchScript、Torch.export路线