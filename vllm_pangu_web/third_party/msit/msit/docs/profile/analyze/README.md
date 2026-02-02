# msit profile analyze功能使用指南

## 简介
- 面向推荐场景的性能调优分析工具, 当前仅支持对GE自动融合算子进行分析。

## 使用方法
### 前置条件
- 工具在运行过程中，会调用ATC工具对GE的dump图进行转换，使用前请先确保source了CANN的环境变量。例如: source /usr/local/Ascend/ascend-toolkit/set_env.sh

### 使用入口
在安装好msit-profile工具后，可以直接通过命令行使用。
```bash
msit profile analyze --origin /tmp/op_summary_origin.csv --fused /tmp/op_summary.csv --ops-graph /tmp/ge_proto_00000001_graph_1_Build.txt
```

### 参数说明
  | 参数名                    | 描述                                       | 必选 |
  | ------------------------ | ---------------------------------------- | ---- |
  | --mode            | 配置模式推理的场景，例如单算子或是图模式，当前仅支持图模式。 | 否  |
  | -f, --framework             | 配置模型推理的AI框架，当前仅支持Tensorflow。           | 否  |
  | --origin       | 未开启算子融合时采集的op_summary文件。 | 是  |
  | --fused       | 开启算子融合后采集的op_summary文件。 | 是  |
  | -ops, --ops-graph       | GE（Graph Engine，图引擎）最后的生成图，即经过GE优化、编译后的图，例如ge_proto_xx_Build.txt。如果是多个图文件，可以输入父目录，例如--ops-graph /tmp，应避免和其他模型的图文件放在同一个目录下。 | 是 |
  | -o, --output       | 性能分析结果保存路径, 默认为当前路径。 | 否  |
  | -h, --help               | 工具使用帮助信息。               | 否  |


## 输出件介绍
> 性能分析结果

运行结束后，指定的output路径下生成一个profile_analysis.csv文件，里面记录了融合算子的性能分析结果，当前仅支持自动融合相关的两类融合算子（算子类型为AscBackend和FusedAscBackend）。
- HBM（High Bandwidth Memory）指的是传输数据量，单位为KB。
分析结果有很多列，下面对每一列的含义进行介绍：

  | 列名                    | 描述                                       | 
  | ------------------------ | ---------------------------------------- | 
  | Fuse OpName   | 融合算子的算子名 | 
  | Fuse OpType   | 融合算子的算子类型    | 
  | Origin Ops    | 融合算子对应的原始算子二元组列表，分别记录每个原始算子的算子名称和算子类型   | 
  | Fused Durations(us)   | 融合算子的耗时总和  | 
  | Origin Durations(us)   | 融合前所有单算子耗时总和  | 
  | Time Ratio    | 融合后算子耗时**除以**融合前算子耗时总和  | 
  | Time Difference   | 融合后算子耗时**减去**融合前算子耗时       | 
  | HBMs Difference  | 融合后算子HBM**减去**融合前算子HBM总和  |
  | HBMs Ratio   | 融合后算子HBM**除以**融合前算子HBM总和  | 
  | Fused HBMs(KB) | 融合后算子的输入和输出HBM(单位为KB)  |
  | Origin Duration(us) Each Op   | 每个原始算子的耗时和列表，分别记录每个原始算子的算子名称和对应耗时和     | 
  | Origin HBMs Each Op(KB) | 每个原始算子的HBM列表，分别记录每个原始算子的输入HBM和输出HBM |
  | Origin HBMs Total(KB) | 所有原始算子输入HBM总和和输出HBM总和 |
  | Not Found Origin Op   | 融合前算子中没有采集到profiling信息的算子名称 | 

> ATC转换文件

  运行过程中会调用ATC工具对GE的dump图进行转换，转换后的文件保存在--output参数指定的路径下，文件名和转换前一样，后缀变成‘json’，即：
  ge_proto_00000001_graph_1_Build.txt -> ge_proto_00000001_graph_1_Build.json。