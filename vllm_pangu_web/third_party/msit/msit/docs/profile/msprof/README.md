# msit profile msprof功能使用指南

## 简介
- 面向om类型文件（由onnx等文件转换为的离线模型）在昇腾设备上进行模型推理性能分析。
- 一键式全流程推理工具msit集成了Profiling性能分析工具，用于分析运行在昇腾AI处理器上的APP工程各个运行阶段的关键性能瓶颈并提出针对性能优化的建议，最终实现产品的极致性能。
- Profiling数据通过二进制可执行文件”msprof”进行数据采集，使用该方式采集Profiling数据需确保应用工程或算子工程所在运行环境已安装Toolkit组件包。
- 该工具使用约束场景说明，参考链接：[CANN商用版/约束说明（仅推理场景）](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasprofiling_16_0003.html)

## 工具安装
- 工具安装请见 [msit一体化工具使用指南](https://gitcode.com/Ascend/msit/blob/master/msit/docs/install/README.md)

## 使用方法
### 功能介绍   
#### 使用入口
profile可以直接通过msit命令行形式启动模型推理的性能分析。使用msit benchmark(msit benchmark为msit自带的推理工程，用户只需修改om路径即可进行模型推理的性能分析及数据采集)推理的性能分析的命令如下：
```bash
msit profile msprof --application "msit benchmark -om *.om --device 0" --output <some path>
```
其中，*为OM离线模型文件名；<some path>为路径名称。
得到主要输出结果如下：
```
<some path>
└── profiler
    └── PROF_000001_20231023172400639_NJDOONIBJCPMJGGB
        ├── device_0 # device侧的结果，device_0表示device id 为0的芯片的性能数据
        │   ├── data # 原始性能数据
        │   ├── log # profiling过程的log日志
        │   ├── summary # 性能数据汇总表格
        │   └── timeline # 通过时间轴呈现性能数据
        └── host
            ├── data # host侧的原始数据
            ├── log # profiling过程的log日志
            ├── summary # 性能数据汇总表格
            └── timeline # 通过时间轴呈现性能数据

```
summary 和 timeline中的文件因命令行参数的选择而不同，详情请见[msprof工具使用](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/opdev/optool/atlasopdev_16_00851.html)


#### 参数说明
  | 参数名                    | 描述                                       | 必选   |
  | ------------------------ | ---------------------------------------- | ---- |
  | --application            | 配置为运行环境上app可执行文件，可配置msit自带的benchmark推理程序，application带参数输入，此时需要使用英文双引号将”application”的参数值括起来，例如--application "msit benchmark -om /home/HwHiAiUser/resnet50.om"，用户使用仅需修改指定om路径 | 是    |
  | -o, --output             | 搜集到的profiling数据的存放路径，默认为当前路径下输出output目录                                                                | 否    |
  | --model-execution        | 控制ge model execution性能数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --sys-hardware-mem       | 控制DDR，LLC的读写带宽数据采集开关，可选on或off，默认为on。 | 否    |
  | --sys-cpu-profiling      | CPU（AI CPU、Ctrl CPU、TS CPU）采集开关。可选on或off，默认值为off。                           | 否    |
  | --sys-profiling          | 系统CPU usage及System memory采集开关。可选on或off，默认值为off。 | 否    |
  | --sys-pid-profiling      | 进程的CPU usage及进程的memory采集开关。可选on或off，默认值为off。 | 否    |
  | --dvpp-profiling         | DVPP采集开关，可选on或off，默认值为on | 否    |
  | --runtime-api            | 控制runtime api性能数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --task-time              | 控制ts timeline数据采集开关，可选on或off，默认为on。该参数配置前提是application参数已配置。 | 否    |
  | --aicpu                  | aicpu开关，可选on或off，默认为on。 | 否  |
  | -h, --help               | 工具使用帮助信息。               | 否  |
