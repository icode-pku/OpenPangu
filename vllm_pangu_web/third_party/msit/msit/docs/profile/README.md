# msit profile 使用说明

## 简介
- 面向在昇腾设备上进行模型推理的性能数据采集和分析。

## 工具安装
- 首先需要安装msit工具，工具安装请见 [msit一体化工具使用指南](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)
- 安装好msit之后，还需要安装profile工具
```bash
msit install profile
```

## 功能介绍
### msprof
集成了性能采集分析工具msprof, 用于分析运行在昇腾AI处理器上的APP工程各个运行阶段的关键性能数据。
[msit profile msprof快速入门指南](./msprof/README.md) 

### analyze
支持在推荐场景、图模式推理的背景下，对采集的profiling数据进行分析，输出性能分析报告用于指导模型性能调优。
[性能比对快速入门指南](./analyze/README.md) 