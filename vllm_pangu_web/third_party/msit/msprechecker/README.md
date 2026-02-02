# MindStudio 预检工具

MindStudio 预检工具（MindStudio Prechecker Tool）是一个帮助用户快速部署服务，快速复现基线，快速定位问题的工具。能提供推理前预检，推理中落盘和推理后比对的功能。
- [**预检**](#预检)：
  检测各种可能会影响服务部署或者性能的组件，支持 VLLM-Ascend 和 MindIE 框架，DeepSeek 与其他模型的部署校验，包括但不限于：
  - 通用检查：检测 CPU 高性能是否开启，透明大页状态，是否为虚拟机，内核版本或昇腾驱动版本是否过低等
  - PD 混部场景：检测环境变量，检测部署服务的 config.json 字段是否合理，检测均通过百分百可以部署成功
  - 单机、双机 PD 分离场景：检测 `conf` 和 `deployment` 目录下的配置文件，确保用户需要修改的字段会进行二次确认
  - 大 EP 场景：检测 `user_config.json` 和 `mindie_env.json` 字段是否合理，检测均通过百分百可以部署成功
  - 模型检查：检测 config.json 中的 `transformers_version` 字段是否小于当前机器的 `transformers` 版本、`torch_dtype` 是否符合当前模型要求等
  - 网络检查：根据 `rank table`，检测各机器芯片之间的连通性，检测各机器相互之间能否 ping 通等
  - CPU、NPU 压测：检测当前机器上是否存在某个核的算力明显低于其他核的情况
- [**落盘**](#落盘)：
  收集各种环境结果并保存到指定路径中，包括但不限于：
  - `env`：环境变量信息
  - `conf`：配置文件字段全量收集，如 `user_config.json` 等
  - `sys`：系统信息收集，如内核版本等
- [**比对**](#比对)：
  根据不同机器落盘结果进行比对，便于快速发现差异点

## 版本配套关系
| MindStudio 预检工具 |     VLLM     |     MindIE     |
|:-------------:|:------------:|:--------------:|
|     依赖版本      | v0.9.1 | ≥ MindIE 2.2.RC1 |

# 环境要求
  - Python 版本要求 >= 3.7
  - 第三方依赖包括：`psutil`, `pyyaml` 和 `importlib_metadata`
  - 相关检测项支持：
    - 机器：`800I A2`, `800I A3` 和 `G8600`
    - 框架：`MindIE`, `VLLM-Ascend`
    - 模型：`DeepSeek`，非 `DeepSeek` 模型

# 安装预检工具
**以下方式选择一种即可**

> **注意**：如果当前用户不是 root，请在安装前执行 `umask 0027`，否则会出现一些安全防护的权限问题，导致使用不便

- **PyPI 安装（推荐）**
  ```bash
  pip install msprechecker
  ```
- **离线安装**
  - 在能够访问网络的机器上，访问 [PyPI 官方源](https://pypi.org/project/msprechecker/#files)
  - 左侧单击 `Download files`，随后单击 `Built Distribution` 下方链接进行下载，如下图所示：

    ![image](./pics/image.png)
  - 下载完成后，上传到服务器中
  - 假设 wheel 包存放路径为 `whl_path`，输入下列命令进行安装
    ```bash
    pip install whl_path
    ```
  - 终端输入 `msprechecker` 校验是否安装成功
- **源码安装**
  在能够访问网络的机器上，执行下列命令即可
  ```bash
  git clone https://gitcode.com/Ascend/msit.git
  pip install -e msit/msprechecker
  ```

# 预检
在部署 PD 分离场景和 PD 混部场景时，大量的配置文件需要用户手动修改，极易出错，且难以定位。预检工具支持 PD 混部和 PD 分离场景的各个配置文件的校验，对于异常的配置，工具通过终端提示错误内容，配置文件会直接展示需要关注的行号，提高部署效率。

工具支持 MindIE 和 VLLM-Ascend 框架（v0.9.1-dev）的预检，以下分不同框架进行介绍。

## MindIE 框架
MindIE 框架不同场景的部署需要修改不同的配置文件，因此以下介绍按照场景进行区分。

### PD 混部场景

```bash
msprechecker precheck --mies-config-path <mindie-service-config>
```

其中，`<mindie-service-config>` 为 `mindie-service` 的配置文件 `config.json`，通常路径为 `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`

混部校验针对 DeepSeek 模型有单独的要求，如果 `<mindie-service-config>` 文件中的 `modelWeightPath` 路径存在，且该目录下的 `config.json` 中的 `model_type` 字段是 `deepseek` 开头，那么工具会触发 DeepSeek 模型混部校验。如果该路径不存在，用户仍可以通过 `--weight-dir` 将权重目录传入，工具会判断该目录下的 `config.json` 中的 `model_type` 字段是否为 `deepseek` 开头，从而触发 DeepSeek 相关校验。

预检工具会对当前终端和配置文件字段进行校验，校验等级分为 `NOK`, `WARNING` 和 `RECOMMEND`，如果当前终端存在环境变量不符合最佳配置要求，则会在当前目录下输出 `msprechecker_env.sh` 文件，便于一键更改配置。校验等级可以通过 `--severity-level [low|medium|high]` 来进行对应更改

> **注意**：部分环境变量需要用户**自行进行确认**，工具无法自动推荐最佳值。比如环境变量 `RANK_TABLE_FILE`，需要用户确认其不仅路径存在，且符合 *rank table* 规范。这种环境变量无法直接通过 source 来进行一键修改。<br><br>
另外，如果是多机混部 PD 场景，则需要在多台机器上 <strong>手动执行</strong> 预检命令，工具 <strong>不支持</strong> 一台机器控制多台机器执行 
> 
> 只有 PD 混部场景才会校验环境变量配置，MindIE 其他部署模式都是通过 k8s 的方式进行 pod 中的环境变量设置，校验配置文件即可

### 单机、多机 PD 分离场景

```bash
msprechecker precheck --scene <deploy-mode> --config-parent-dir <kubernetes-dir>
```

其中，`<deploy-mode>` 支持两种选项：
- `pd_disaggregation`: 多机 PD 分离场景
- `pd_disaggregation_single_container`: 单机 PD 分离场景

`<kubernetes-dir>` 为官方要求用户需要修改的 `conf/*.json` 和 `deployments/*.yaml` 的父目录，一般目录命名为 `kubernetes_deploy_scripts`。工具会对父目录所有官方指南中期望用户修改的文件进行确认，如果存在字段填写问题，会进行输出。

如果只提供了 `<deploy-mode>`，但是没有提供 `<kubernetes-dir>`，工具会进行如下提示：

```bash
$ msprechecker precheck --scene pd_disaggregation
Passing '--scene' without providing '--config-parent-dir' will not take any effect!
```

> **注意**：目前多机 PD 分离仅支持 800I A2 机器

### 大 EP 场景

```bash
msprechecker precheck --user-config-path <user-config> --mindie-env-path <mindie-env>
```

其中 `<user-config>` 为大 EP 场景下需要手动配置的 `user_config.json` 文件；`<mindie-env>` 为管理 Pod 拉起之后的环境变量文件 `mindie_env.json`。用户通过命令行的方式将路径传入即可触发校验。

## VLLM-Ascend 框架
VLLM-Ascend 框架的配置通过命令行参数进行管理。预检工具提供针对不同部署场景的专项检查，确保环境配置的正确性和最佳性能。

### 通用场景
针对 VLLM-Ascend 的单机、多机 PD 场景，PD 混部场景

```bash
msprechecker precheck --scene vllm
```
相比默认场景新增校验:
- 环境相关校验：
  - OpenMP 多线程绑核配置性能
  - HCCL 通信缓存等
  - LD_PRELOAD 中有没有生效 `Jemalloc`
- 系统相关校验
  - 新增 `Jemalloc` 是否安装判断

### PD 分离场景

```bash
msprechecker precheck --scene vllm,ep
```

相比 VLLM-Ascend **通用场景** 新增校验:
- 昇腾相关校验：
  - 校验当前机器的昇腾驱动版本是否大于 25.0
- 环境相关校验：
  - 新增 PD 分离场景相关校验

## 默认场景

```bash
msprechecker precheck
```

不带任何参数的运行预检模式，会触发三个默认校验：
- 系统相关校验：
  - 校验当前机器的 CPU 高性能模式是否开启
  - 校验当前机器的透明大页是否开启
  - 校验当前机器的 Linux 内核版本是否大于 5.10
- 昇腾相关校验：
  - 校验当前机器的昇腾驱动版本是否大于 24.1

## 其他校验

```bash
msprechecker precheck --hardware --threshold <threshold> --weight-dir <weight-dir> --rank-table-path <rank-table-path>
```

上述命令除了**包含默认场景校验项**之外，额外包含四个校验：
- `--hardware` 触发 CPU、NPU 压测；`--threshold` 用于设置压测筛选阈值，便于自定义筛选有问题的 CPU 核或者 NPU 卡
- `<weight-dir>` 为权重目录，用于检测权重目录下的 `config.json` 的字段
- `<rank-table-path>` 为 *rank table* 路径，用于触发 `hccn_tool` 相关的校验，以及多机之间的连通性测试，并且会对配置文件字段进行检查

> 须知：
> 不同框架和硬件型号（如 MindIE 与 VLLM-Ascend，或 800I A2 与 800I A3）所使用的 *rank table* 格式存在差异。
> 工具通过 `--scene` 参数来识别并解析对应框架的 *rank table* 格式。若未指定 `--scene`，工具将**默认**按 MindIE 框架的格式进行解析。

# 落盘
推理过程中，如果出现 **异常** 或者 ​**性能不及预期**​，可以使用 ​**落盘** 功能​，将环境相关信息进行落盘，方便后续比对。推理结束后，性能预检工具支持比对推理中落盘的环境变量和配置项，帮助快速发现可能影响性能的差异点，实现问题快速定位

> 注意：目前落盘功能不具备落盘多机 PD 分离、单机 PD 分离配置文件的能力

使用落盘功能只需在终端中输入 `msprechecker dump --output-path <output-path>`，其中 `<output-path>` 为用户指定的输出路径；如不指定，则默认保存在当前目录下，名为 `./msprechecker_dumped.json`。示例如下：
```bash
$ msprechecker dump
Error occurred while collecting 'hccl': 宿主机上没有找到 'hccn_tool' 命令
All information has been saved in: './msprechecker_dumped.json'. You can use '--output-path' to specify the save location.
What's Next?
        You may now use 'msprechecker compare' to compare two or more dumped files for discrepancies!
```
其中的 *Error* 字段表示在落盘过程中出现的一些问题，不会影响其他数据的落盘。上述错误表明在落盘 HCCL 相关数据时，由于当前宿主机上没有 `hccn_tool` 命令，因此不会收集。

随后工具会提示落盘位置，和如何更改落盘位置，以及后续可以如何操作落盘文件。

# 比对
在进行比对前，请确保使用预检工具落盘两个或多个文件，单个文件无法比对。如果只传入单个文件路径，会出现报错提示：

```bash
$ msprechecker compare msprechecker_dumped.json
You need two or more files to compare!
```

假设我在家里落盘了一个基线文件 `baseline.json`，在客户现场落盘了一个现场文件 `dumped.json`，那么我可以将这两个文件放在当前目录下执行如下命令进行比对
```bash
msprechecker compare baseline.json dumped.json
```
预检工具会将不同内容，分段进行展示，如果没有差异会提示 `There is no difference found.` 有差异内容会展示如下：

```bash
============================= SYSTEM DIFF REPORT ==============================
{
  "high_performance": {
    "baseline.json": false,
    "dumped.json": true
  },
  "transparent_hugepage": {
    "baseline.json": "madvise",
    "dumped.json": "always"
  },
  "virtual_machine": {
    "baseline.json": true,
    "dumped.json": false
  }
}
```

# 自定义检查项配置
在进行预检时，工具支持自定义配置校验项。用户可以传入自定义的规则 yaml 文件，通过 `--custom-config-path` 参数传递给工具从而完成自定义校验。

假设需要校验 `a.b` 的值是否符合要求，那么自定义配置语法如下
```sh
a:
  b:
    expected:
      type: eq
      value: 1 + 2
    reason: a.b 的值应该等于 3
    severity: high
```
上述配置表示，`a.b` 的值应该等于 3，其严重程度为 high，如果该字段不符合预期，会显示 `[NOK]`

目前，
- `type` 支持：`eq`, `lt`, `le`, `gt`, `ge`, `ne` 或者 `==`, `<`, `<=`, `>`, `>=`, `!=`
- `value` 支持：`+`, `-`, `*`, `/`, `//`, `**`, `()`，还支持字段引用 `${}` 和版本符号 `Version{}`
- `reason` 支持任意字符串
- `severity` 支持：`low`, `medium`, `high`，不填写默认 `high`。其中，`low` 显示为 `[RECOMMEND]`；`medium` 显示为 `[WARNING]`；`high` 显示为 `[NOK]`

> 注意：目前自定义配置文件只支持用户自定义配置 **环境变量** 相关校验，且 **不支持 PD 单机分离和 PD 多机分离的场景**

# 字段引用
对于比较嵌套较深的配置文件，遇到不同字段相互关联的场景时，创建校验规则是一个挑战。预检工具支持 **字段引用** 语法，允许用户通过 `${}` 的语法来引用其他位置的字段值。比如，有如下配置文件
```json
{
  "a": {
    "b": "value of b",
    "c": "value of c"
  }
}
```
如果希望 `a.b` 的值和 `a.c` 的值相等，则校验规则如下：
```yaml
a:
  b:
    expected:
      type: eq
      value: ${.c}
    reason: a.b 的值等于 a.c
    severity: high
  c:
  expected:
      type: eq
      value: ${a.b}
    reason: a.c 的值等于 a.b
    severity: high
```
其中 `${.c}` 是相对引用, `${a.b}` 是绝对引用；对于大型嵌套的配置文件尤其好用。

# 参数列表
预检工具于 2025年8月1日迎来了一次大规模重构，参数采用 `-` 分割而不是 `_` 分割，方便用户输入（不再需要额外的 `shift` 键参与）。部分不常用参数移除，其他的遗留参数进行了向后兼容，但是工具会在使用时给出 deprecation 提示。
  
子功能包括 `precheck` / `dump` / `compare`，用户可以通过 `msprechecker -h` 获取子功能列表，以及 `msprechecker {子功能} -h` 获取对应子功能的参数列表

## precheck 参数

### Legacy 参数
以下参数为遗留参数，为了向后兼容

| 参数名                          | 参数描述                                                                                        | 是否必选                                    |
| ------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------- |
| -ch {...}, --checkers {...}     | *字符串列表值，可选值 basic,hccl,model,hardware,all，指定检查项，可指定多个，all 表示全部        | 否，默认值 basic                            |
| -service, --service_config_path | 字符串值，MINDIE service 路径或 config json 文件路径，优先级高于环境变量的 MIES_INSTALL_PATH 值 | 否，默认使用环境变量的 MIES_INSTALL_PATH 值 |
| -user, --user_config_path       | 字符串值，json 文件，k8s user_config.json 文件，不指定则不检查                                    | 否，默认 None                               |
| --mindie_env_config_path        | 字符串值，json 文件，k8s mindie_env.json 文件，不指定则不检查                                     | 否，默认 None                               |
| -ranktable, --ranktable_file    | 字符串值，json 文件，手动指定 ranktable 文件，优先级高于环境变量的 RANKFILETABLE                | 否，默认使用环境变量的 RANKFILETABLE 值     |
| --weight_dir   |  模型权重目录路径        | 否，默认使用 config.json 中的 `modelWeightPath` 字段路径   |
| -add, --additional_checks_yaml  | 字符串值，yaml 文件，额外的自定义配置项，指定后将覆盖默认检查项中的值                           | 否，默认 None                               |
| -d, --dump_file_path | 字符串值，指定 dump 数据的保存路径 | 否，默认为 msprechecker_dumped.json |

- `basic` 表示检查环境变量以及基本系统信息
- `hccl` 表示检查 NPU 之间通过 hccl 连接的状态
- `model` 表示检查或比对模型大小以及 sha256sum 值
- `hardware` 表示检查 CPU / NPU 计算能力，以及网络连接状态
- `all` 表示检查全部

除了 Legacy 参数外，precheck 功能提供以下额外参数：

### PD 分离选项
| 参数名         | 参数描述                                             | 是否必选                       |
| -------------- | ---------------------------------------------------- | ------------------------------ |
| --scene | 指定框架和 PD 部署策略。该参数有两种输入模式，一种是为了向前兼容，一种是后续推荐输入方式：<ul><li>【向前兼容】如果输入 `pd_disaggregation`, `pd_disaggregation_single_container` 则表明框架为 `mindie`，场景依次为单机 PD 分离或者多机 PD 分离；如果输入 `mindie` 或者 `vllm`，则表明框架依次为 `mindie` 或者 `vllm`，PD 部署策略为 PD 混部</li><li>【推荐】如果输入 `<framework>,<deploy-mode>`，逗号前的内容会被解析为框架，逗号后的内容会被解析为 PD 部署策略。<ul><li>框架：目前支持选择 `mindie` 或者 `vllm`</li><li>PD 部署策略：目前支持 `pd_disaggregation`, `pd_disaggregation_single_container`, `ep` 或者 `pd_mix`</li></ul></li></ul><br>注意：如果 PD 部署策略为单机 PD 分离或者多机 PD 分离，需要同时提供 `--config-parent-dir` 方能进行校验。 | 否 |
| --user-config-path | 指定大 EP 场景下的 `user_config.json` 路径。其对应的 Legacy 参数为 `-user, --user_config_path` | 否 |
| --mindie-env-path | 指定大 EP 场景下的 `mindie_env.json` 路径。其对应的 Legacy 参数为 `--mindie_env_config_path` | 否 |
| --config-parent-dir | 指定 PD 分离场景下，所有配置文件的父目录，通常名为 `kubernetes_deploy_scripts`。这个参数需要和 `--scene` 一起使用 | 否 |

### PD 混部选项
| 参数名         | 参数描述                                             | 是否必选                       |
| -------------- | ---------------------------------------------------- | ------------------------------ |
| --mies-config-path | 指定 PD 混部模式所需要修改的 `config.json` 路径，通常路径为 `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`。 其对应的 Legacy 参数为 `-service, --service_config_path` | 否 |

### Network 选项
| 参数名         | 参数描述                                             | 是否必选                       |
| -------------- | ---------------------------------------------------- | ------------------------------ |
| --rank-table-path | 指定 *rank table* 路径，用于进行 `hccn_tool` 测试、多机连通性测试和文件字段检查。其对应的 Legacy 参数为 `-ranktable, --ranktable_file` | 否 |

### Model 选项
| 参数名         | 参数描述                                             | 是否必选                       |
| -------------- | ---------------------------------------------------- | ------------------------------ |
| --weight-dir | 指定模型权重目录路径，用于进行权重目录下的 `config.json` 检查。其对应的 Legacy 参数为 `--weight_dir` | 否 |

### 压力测试选项
| 参数名         | 参数描述                                             | 是否必选                       |
| -------------- | ---------------------------------------------------- | ------------------------------ |
| --hardware | 是否开启 CPU、NPU 的压测，输入则开启。默认不开启。其对应的 Legacy 使用方式为 `-ch hardware` | 否，默认  `False` |
| --threshold | int，0-100 之间，单位为 %，用于控制压测筛选阈值，某个核的计算时间超过平均值的比例大于该阈值，则被认定为异常 | 否，默认 20 |

### 其他校验选项
| 参数名         | 参数描述                                             | 是否必选                       |
| -------------- | ---------------------------------------------------- | ------------------------------ |
| --custom-config-path | 用户自定义规则路径。其对应的 Legacy 参数为 `-add, --additional_checks_yaml` | 否 |
| -l, --severity-level | 用于指定校验严重度等级，只可能选择 `low`, `medium` 或者 `high`。其中 `low` 全部显示，`high` 只显示 `NOK` 项 | 否，默认 `low` |

## dump 额外参数

### Legacy 参数
以下参数为遗留参数，为了向后兼容

| 参数名                          | 参数描述                                                                                        | 是否必选                                    |
| ------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------- |
| -ch {...}, --checkers {...}     | *字符串列表值，可选值 basic,hccl,model,hardware,all，指定检查项，可指定多个，all 表示全部        | 否，默认值 basic                            |
| -service, --service_config_path | 字符串值，MINDIE service 路径或 config json 文件路径，优先级高于环境变量的 MIES_INSTALL_PATH 值 | 否，默认使用环境变量的 MIES_INSTALL_PATH 值 |
| -user, --user_config_path       | 字符串值，json 文件，k8s user_config.json 文件，不指定则不检查                                    | 否，默认 None                               |
| --mindie_env_config_path        | 字符串值，json 文件，k8s mindie_env.json 文件，不指定则不检查                                     | 否，默认 None                               |
| -ranktable, --ranktable_file    | 字符串值，json 文件，手动指定 ranktable 文件，优先级高于环境变量的 RANKFILETABLE                | 否，默认使用环境变量的 RANKFILETABLE 值     |
| --weight_dir   |  模型权重目录路径        | 否，默认使用 config.json 中的 `modelWeightPath` 字段路径   |
| -add, --additional_checks_yaml  | 字符串值，yaml 文件，额外的自定义配置项，指定后将覆盖默认检查项中的值                           | 否，默认 None                               |
| -d, --dump_file_path | 字符串值，指定 dump 数据的保存路径 | 否，默认为 msprechecker_dumped.json |

- `basic` 表示检查环境变量以及基本系统信息
- `hccl` 表示检查 NPU 之间通过 hccl 连接的状态
- `model` 表示检查或比对模型大小以及 sha256sum 值
- `hardware` 表示检查 CPU / NPU 计算能力，以及网络连接状态
- `all` 表示检查全部

除了 Legacy 参数外，dump 功能提供以下额外参数：

| 参数名               | 参数描述                           | 是否必选                                            |
| -------------------- | ---------------------------------- | --------------------------------------------------- |
| --output-path | 指定落盘路径。其对应的 Legacy 参数为 `-d, --dump_file_path` | 否，默认 `msprechecker_dumped.json`
|

### 其他参数
| 参数名               | 参数描述                           | 是否必选                                            |
| -------------------- | ---------------------------------- | --------------------------------------------------- |
| --filter | 是否只落盘跟昇腾相关的环境变量。开启该选项则 **不会** 落盘如 `LD_LIBRARY_PATH` 的环境变量，因为与昇腾研发的环境变量无关  | 否，默认 `False`
| --user-config-path | 额外落盘大 EP 场景下的 `user_config.json` 路径。其对应的 Legacy 参数为 `-user, --user_config_path` | 否 |
| --mindie-env-path | 额外落盘大 EP 场景下的 `mindie_env.json` 路径。其对应的 Legacy 参数为 `--mindie_env_config_path` | 否 |
| --mies-config-path | 额外落盘 PD 混部模式所需要修改的 `config.json` 路径，通常路径为 `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`。 其对应的 Legacy 参数为 `-service, --service_config_path` | 否 |
| --rank-table-path | 额外落盘 *rank table* 文件。其对应的 Legacy 参数为 `-ranktable, --ranktable_file` | 否 |
| --weight-dir | 额外落盘模型权重目录下的 `config.json` 和所有 `*.safetensors` 权重 sha256sum 哈希值。其对应的 Legacy 参数为 `--weight_dir` | 否 |
| --chunk-size | int，单位为 MB，指定在计算权重哈希时，每次读取文件的大小，只支持 32, 64, 128 或者 256。默认为 32 | 否 |

 
## compare 参数
compare 功能**只有**如下参数：

| 参数名                      | 参数描述                                                 | 是否必选       |
| --------------------------- | -------------------------------------------------------- | -------------- |
| FILE             | **位置参数**，依次指定工具 dump 的多份数据路径，路径之间空格分割         | 是，且应为两个或两个以上 |

## 废弃参数
| 参数名         | 参数描述                                             | 废弃原因                       |
| -------------- | ---------------------------------------------------- | ------------------------------ |
| -blocknum, --sha256_blocknum | int 值，计算模型权重 sha256sum 值时的采样块数，值越大采样越多，计算速度越慢 | 预检对权重哈希进行校验，由于缺乏标准，没有意义，目前将此功能转移到落盘模式中，通过 `--chunk-size`，将 `--weight-dir` 下的权重的哈希值进行落盘，方便比对 |
| -l {...}, --log_level {...}     | 日志级别，可选值 debug,info,warning,error,fatal,critical                                        | 预检工具不需要控制日志级别，只需要控制校验严重性级别，目前通过 `--severity-level` 进行控制，该参数废弃                              |
| -s, --save_env | 字符串值，指定环境变量需要优化时，输出的 sh 文件路径	| 预检工具在预检过程需要用户指定输出路径很奇怪，且随着校验项逐步增多，工具可能无法再提供一键修改全部环境变量。并且输出一个文件进行 source 存在潜在安全风险，待后续进一步评估，目前先移除指定落盘位置功能 |
