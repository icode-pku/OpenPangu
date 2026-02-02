# vLLM 服务化性能采集工具

在推理服务过程中，我们有时需要监控推理服务框架的内部执行流程以定位性能问题。通过采集关键流程的起止时间、识别关键函数或迭代、记录关键事件并捕获多种类型的信息，可以快速定位性能瓶颈。

本部分将指导你如何采集 vllm-ascend 的服务化框架性能数据以及算子性能数据，覆盖从准备、采集、解析到结果展示的完整流程，帮助你快速上手性能采集工具。

## 环境准备
根据不同版本需求，[vLLM Ascend installation](https://vllm-ascend.readthedocs.io/en/latest/installation.html) 成功启动推理服务。

## 安装 vllm_profiler 组件

- 方法1：使用 pip 安装 `msserviceprofiler` 包的稳定版本
  
  ```bash
  pip install msserviceprofiler==1.2.2
  ```

- 方法2：源码安装

  ```bash
  git clone https://gitcode.com/Ascend/msit.git
  cd msit/msserviceprofiler
  pip install -e .
  
  cd -
  git clone -b msserviceprofiler_dev https://gitcode.com/ascend/msit.git msserviceprofiler_dev
  export PYTHONPATH=$PWD/msserviceprofiler_dev/msserviceprofiler/:$PYTHONPATH
  ```


## 快速开始

### 1. 准备采集
在启动服务之前，请设置环境变量 `SERVICE_PROF_CONFIG_PATH` 指定需要加载的性能分析配置文件，并设置环境变量 `PROFILING_SYMBOLS_PATH` 来指定需要导入的符号的 YAML 配置文件。之后，根据您的部署方式启动 vLLM 服务。

```bash
cd ${path_to_store_profiling_files}
# 设置环境变量
export SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json
export PROFILING_SYMBOLS_PATH=service_profiling_symbols.yaml

# 启动 vLLM 服务
vllm serve Qwen/Qwen2.5-0.5B-Instruct &
```

其中 `ms_service_profiler_config.json` 为采集配置文件。若指定路径下不存在该文件，将自动生成一份默认配置。若有需要，可参照[采集配置文件说明](#2-采集配置文件说明)章节提前进行自定义配置。

`service_profiling_symbols.yaml` 为需要导入的埋点配置文件。你也可以选择不设置环境变量 `PROFILING_SYMBOLS_PATH` ，此时将使用默认的配置文件；若你指定的路径下不存在该文件，系统同样会在你指定的路径生成一份配置文件以便后续修改。可参考[点位配置文件说明](#3-点位配置文件说明)一节进行自定义。

### 2. 开启采集
将配置文件`ms_service_profiler_config.json`中的 `enable` 字段由 `0` 修改为 `1`，即可开启性能数据采集的开关，可以通过执行下面sed指令完成采集服务的开启：

```bash
sed -i 's/"enable":\s*0/"enable": 1/' ./ms_service_profiler_config.json
```

### 3. 发送请求
根据实际采集需求选择请求发送方式：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json"  \
    -d '{
         "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Beijing is a",
        "max_tokens": 5,
        "temperature": 0
}' | python3 -m json.tool
```

### 4. 解析数据

```bash
# xxxx-xxxx 为采集工具根据 vLLM 启动时间自动创建的存放目录
cd /root/.ms_server_profiler/xxxx-xxxx

# 解析数据
msserviceprofiler analyze --input-path=./ --output-path output
```

如果你使用的是源码安装的形式，需要用下面的命令进行解析：

```bash
# 解析数据
python -m ms_service_profiler.parse --input-path=./ --output-path output
```

### 5. 查看结果

解析完成后，`output` 目录下会生成下面表格中列出的交付件。依据安装方式的不同，输出件内容会有差异：

| 输出件 | 说明 | pip安装 | 源码安装 |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------:|:------:|
| `chrome_tracing.json` | Chrome 追踪格式数据，可在 [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html) 中打开 | ✅ | ✅ |
| `profiler.db` | 数据库格式的性能数据 | ✅ | ✅ |
| `request.csv` | 请求相关数据 | ✅ | ✅ |
|`request_summary.csv` | 请求总体统计指标 | ✅ | ❌ |
| `kvcache.csv` | KV Cache 相关数据 | ✅ | ✅ |
| `batch.csv` | 批次调度相关数据 | ✅ | ✅ |
| `batch_summary.csv` | 批次调度总体统计指标 | ✅ | ❌ |
| `service_summary.csv` | 服务化维度总体统计指标。 | ✅ | ❌ |
---

## 附录

### 1. 版本支持情况

  |  配套CANN版本   | vLLM-ascend V0 | vLLM-ascend V1 |
  |:-------:|:----:|:----:|
  | 8.2.RC1 |  /  | v0.11.0.RC0 |
  | 8.2.RC1 |  /  | v0.10.2.RC1 |
  | 8.2.RC1 |  /  | v0.10.1.RC1 |
  | 8.2.RC1 |  /  | v0.10.0.RC1 |
  | 8.2.RC1 |  /  | v0.9.2.RC1 |
  | 8.2.RC1 |  v0.9.1  | v0.9.1 |
  | 8.1.RC1 |  v0.8.5.RC1  | / |
  | 8.1.RC1 |  v0.8.4   | / |
  | 8.0.RC3 |  v0.6.3   | / |

### 2. 采集配置文件说明

采集配置文件用于控制性能数据采集的参数与行为。

#### 配置文件格式

配置文件为 JSON 格式，主要参数如下：

| 参数 | 说明 | 是否必选 |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| enable | 是否开启性能数据采集的开关：<br />0：关闭<br />1：开启<br />默认值：0 | 是 |
| prof_dir | 采集到性能数据的存放路径，支持用户自定义。<br />默认值：`$HOME/.ms_service_profiler` | 否  |
| profiler_level | 数据采集等级。默认值为"INFO"，指普通级别的性能数据。 | 否 |
| host_system_usage_freq | CPU和内存系统指标采集频率，默认关闭不采集。范围整数1~50，单位hz，表示每秒采集的次数。设置为-1时关闭采集该指标。<br />说明：开启该功能可能占用较大内存 | 否 |
| npu_memory_usage_freq | NPU Memory使用率指标的采集频率，默认关闭不采集。范围整数1~50，单位hz，表示每秒采集的次数。设置为-1时关闭采集该指标。<br />说明：开启该功能可能占用较大内存 | 否 |
| acl_task_time | 开启采集算子下发耗时、算子执行耗时数据的开关，取值为：<br />0：关闭。默认值，配置为0或其他非法值均表示关闭。<br />1：开启。该功能开启时调用aclprofCreateConfig接口的ACL_PROF_TASK_TIME_L0参数。<br />2：开启基于MSPTI接口的数据落盘。该功能开启时调用MSPTI接口进行性能数据采集，需要配置如下环境变量：`export LD_PRELOAD=$ASCEND_TOOLKIT_HOME/lib64/libmspti.so` | 否 |
| acl_prof_task_time_level | 设置性能数据采集的Level等级和时长，取值为：<br />L0：Level0等级，表示采集算子下发耗时、算子执行耗时数据。与L1相比，由于不采集算子基本信息数据，采集时性能开销较小，可更精准统计相关耗时数据。<br />L1：Level1等级，采集AscendCL接口的性能数据，包括Host与Device之间、Device间的同步异步内存复制时延；采集算子下发耗时、算子执行耗时数据以及算子基本信息数据，提供更全面的性能分析数据。<br />time：采集时长，取值范围为1~999的正整数，单位s。<br />默认未配置本参数，表示采集L0数据，且采集到程序执行结束。配置其他非法值时取默认值。<br />采集的Level等级和时长可同时配置，例如`"acl_prof_task_time_level": "L1,10"`。 | 否 |
| api_filter | 对性能数据进行过滤，配置该参数可自定义采集配置的API性能数据，例如传入"matmul"会落盘所有API数据中name字段包含matmul的性能数据。str类型，区分大小写，多个不同的筛选目标用"；"隔开，默认为空，表示落盘所有数据。<br />仅当`acl_task_time`参数值为2时生效。 | 否 |
| kernel_filter | 对性能数据进行过滤，配置该参数可自定义采集配置的kernel性能数据，例如传入"matmul"会落盘所有kernel数据中name字段包含matmul的性能数据。str类型，区分大小写，多个不同的筛选目标用"；"隔开，默认为空，表示落盘所有数据。<br />仅当`acl_task_time`参数值为2时生效。 | 否 |
| timelimit | 设置服务化性能数据采集的时长，配置该参数后，采集进程将在运行指定的时间后自动停止，取值范围为0~7200的整数，单位s，默认值0（表示不限制采集时间） | 否 |
| domain | 设置采集指定domain域下的性能数据，减少采集数据量。输入参数为字符串格式，英文分号作为分隔符，区分大小写，例如："Request; KVCache"。<br />默认为空，表示采集当前所有domain域内性能数据。<br />当前已有domain域为：Request、KVCache、ModelExecute、BatchSchedule、Communication。<br />说明：<br />若指定domain域不全，采集数据不满足解析输出件生成时，会有告警提示。[查看domain域与解析结果对照表](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/Profiling/mindieprofiling_0009.html) | 否 |

#### 配置示例

```json
{
  "enable": 1,
  "prof_dir": "vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "acl_prof_task_time_level": "",
  "timelimit": 0
}
```

---

### 3. 点位配置文件说明

点位配置文件用于定义需要采集的函数/方法，支持灵活配置与自定义属性采集。

#### 3.1. 文件命名与加载

- 默认加载路径：`~/.config/vllm_ascend/service_profiling_symbols.MAJOR.MINOR.PATCH.yaml`（适用于 vLLM-ascend 框架且文件名随已安装的 vllm 版本变化）
- 备用加载路径：`工具安装路径/msserviceprofiler/vllm_profiler/config/service_profiling_symbols.yaml`

如需自定义采集点，推荐通过设置环境变量`PROFILING_SYMBOLS_PATH`，将一份点位配置文件复制到工作目录进行修改使用。

#### 3.2. 配置字段说明

| 字段 | 说明 | 示例 |
|:-----:|:-----|:-----|
| symbol | Python 导入路径 + 属性链 | `"vllm.v1.core.kv_cache_manager:KVCacheManager.free"` |
| handler | 处理函数类型 | `"timer"`（默认）或 `"pkg.mod:func"`（自定义） |
| domain |埋点域标识 | `"KVCache"`, `"ModelExecute"` |
| name | 埋点名称 | `"EngineCoreExecute"` |
| min_version | 最高版本约束 | `"0.9.1"` |
| max_version | 最低版本约束 | `"0.11.0"` |
| attributes | 自定义属性采集 | 只支持 `"timer"` handler。详见下方自定义属性采集机制 |

#### 3.3. 配置示例

- 示例 1：自定义处理函数

```yaml
- symbol: vllm.v1.core.kv_cache_manager:KVCacheManager.free
  handler: vllm_profiler.config.custom_handler_example:kvcache_manager_free_example_handler
  domain: Example
  name: example_custom
```

- 示例 2：默认计时器

```yaml
- symbol: vllm.v1.engine.core:EngineCore.execute_model
  domain: ModelExecute
  name: EngineCoreExecute
```

- 示例 3：版本约束

```yaml
- symbol: vllm.v1.executor.abstract:Executor.execute_model
  min_version: "0.9.1"
  # 未指定 handler -> 默认 timer
```

#### 3.4. 自定义属性采集机制

`attributes` 字段支持灵活的自定义属性采集，可对函数参数与返回值进行多种操作与转换。

##### 基本语法

- 参数访问：直接使用参数名，如 `input_ids`
- 返回值访问：使用 `return` 关键字
- 管道操作：使用 `|` 分隔多个操作
- 属性访问：使用 `attr` 获取对象属性

##### 配置示例

```yaml
- symbol: vllm_ascend.worker.model_runner_v1:NPUModelRunner.execute_model
  name: ModelRunnerExecuteModel
  domain: ModelExecute
  attributes:
  - name: device
    expr: args[0] | attr device | str
  - name: dp
    expr: args[0] | attr dp_rank | str
  - name: batch_size
    expr: args[0] | attr input_batch | attr _req_ids | len
```

##### 表达式说明

1. `len(input_ids)`：获取 `input_ids` 参数的长度。
2. `len(return) | str`：获取返回值长度并转换为字符串（等价于 `str(len(return))`）。
3. `return[0] | attr input_ids | len`：获取返回值第一个元素的 `input_ids` 属性长度。

##### 支持的表达式类型

- 基础操作：`len()`, `str()`, `int()`, `float()`
- 索引访问：`return[0]`, `return['key']`
- 属性访问：`return | attr attr_name`
- 管道组合：多个操作通过 `|` 连接

##### 高级示例

```yaml
attributes:
  # 获取张量形状
  - name: tensor_shape
    expr: input_tensor | attr shape | str
  
  # 获取字典中的特定值
  - name: batch_size
    expr: kwargs['batch_size']
  
  # 条件表达式（需要自定义处理函数支持）
  - name: is_training_mode
    expr: training | bool
  
  # 复杂的数据处理
  - name: processed_data_len
    expr: data | attr items | len | str
```

#### 3.5. 自定义处理函数

当 `handler` 字段指定自定义处理函数时，该函数需满足以下签名：

```python
def custom_handler(original_func, this, *args, **kwargs):
    """
    自定义处理函数
    
    Args:
        original_func: 原始函数对象
        this: 调用对象（对于方法调用）
        *args: 位置参数
        **kwargs: 关键字参数
    
    Returns:
        处理结果
    """
    # 自定义处理逻辑
    pass
```

若自定义处理函数导入失败，系统会自动回退至默认计时器模式。
