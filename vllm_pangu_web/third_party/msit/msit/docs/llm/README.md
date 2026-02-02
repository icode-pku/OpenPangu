# 大模型推理精度工具（Large Language Model Debug Tool）

![精度问题工作流](/msit/docs/llm/image/acc-workflow.png)

## 简介

目前昇腾大模型推理框架主要有 [**MindIE**](https://www.hiascend.com/software/mindie) 和 [**torchair**](/msit/docs/glossary/README.md#torchairtorch-图模式)。

开发者在推理开发过程中可能会遇到精度问题，可以使用大模型精度调试工具（Large Language Model Debug Tool）提供的大模型推理数据落盘（dump）和精度定位（compare）功能，帮助开发者快速定位推理开发过程中精度问题，发现根因，提升开发效率。

#### 安装

- msit llm 使用依赖 CANN-toolkit、加速库 ATB和 MindIE-LLM，各依赖库版本要求参照具体[安装说明文档](/msit/docs/install/README.md)。
- **注意**：
- 1. 加速库数据dump仅支持2023/12/05之后的加速库版本；
- 2. 当前Torch的版本为2.1.0，Python的版本至少为3.8.0及以上

```bash
# 源码安装：先下载源码，进到源码目录
pip install ./msit
msit install llm
```

* msit历史版本提供whl包进行安装，可以参考：[历史版本安装](#历史版本安装)

#### 工具列表
> * [BadCase分析使用说明](../llm/工具-大模型精度定位之BadCase分析工具.md)
> * [dump加速库数据使用说明](../llm/工具-DUMP加速库数据使用说明.md)
> * [PyTorch 场景的精度数据采集](../llm/工具-Pytorch场景数据dump.md)
> * [自动比对功能使用说明](../llm/工具-大模型精度比对.md)
> * [精度预检使用说明](../llm/工具-精度预检使用说明.md)
> * [异常检测使用说明](../llm/工具-异常检测使用说明.md)

#### 场景列表
> * [加速库场景-输出Token的logits精度比对](../llm/加速库场景-输出Token的logits精度比对.md)
> * [工具-大模型精度比对](../llm/工具-大模型精度比对.md)
> * [TorchAir场景-整网算子精度比对](../llm/TorchAir场景-整网算子精度比对.md)

## 大模型精度调试步骤


大模型精度调试定位，一般思路是先定位到具体存在精度问题的输入，再从整网到算子，从外到内，从粗到细逐步定位根因，具体定位操作可以视情况调整。一般分为以下步骤：

1. 定位存在精度问题的输入
   1. **Bad Case 分析**: 当数据集评估不理想，需要找到存在精度问题的Bad Case ,可以通过 [**BadCase分析使用说明**](/msit/docs/llm/工具-大模型精度定位之BadCase分析工具.md) 定位。
   2. **输出 token 比对**: 当生成任务遇到输出误差逐渐变大的场景，需要快速识别是第几个token开始出现精度问题，可以通过dump和比对工具识别，具体可以参考 [**加速库场景-输出Token的logits精度比对**](/msit/docs/llm/加速库场景-输出Token的logits精度比对.md)。
2. 定位存在精度问题的算子
   1. **整网算子精度比对**: 当相同输入但是 npu 和 cpu(gpu)输出不一致，可以通过逐层算子比对方式定位到存在精度问题的算子。
      - 概要流程：
        - 需要dump 标杆数据和存在精度问题的数据，llm内提供了多种工具dump数据。
        - 将dump 数据进行自动比对或者手动比对，查找输入误差小，但是输出误差大的算子。
        - 可以由粗到细，由外到内的排查。先dump定位哪个block存在异常，再dump block内部算子数据进行进一步定位。可以减少dump的时间以及磁盘空间占用。
      - 排查流程详细说明文档：
        - [**工具-大模型精度比对**](/msit/docs/llm/工具-大模型精度比对.md)：加速库推理场景，如何定位存在精度问题算子。
        - [**TorchAir场景-整网算子精度比对**](/msit/docs/llm/TorchAir场景-整网算子精度比对.md)：torchair 推理场景，如何定位存在精度问题算子。
      - 相关功能：
        * [**dump加速库数据使用说明**](/msit/docs/llm/工具-DUMP加速库数据使用说明.md)：提供了 dump 加速库的网络结构、算子信息、推理输入输出等信息，支撑后续手动和自动比对、分析工作。
        * [**PyTorch 场景的精度数据采集**](/msit/docs/llm/工具-Pytorch场景数据dump.md)：提供了通过pyTorch框架 使用 GPU/CPU/NPU 在线推理场景的网络结构、算子信息、推理输入输出等信息，支撑后续手动和自动比对、分析工作。
        * [**自动比对功能使用说明**](/msit/docs/llm/工具-大模型精度比对.md)：提供了自动比对功能，比对标杆数据和推理数据之间的误差。
   2. **异常检测**: 定位推理过程中是否存在算子计算溢出、内存踩踏。可以参考 [**异常检测使用说明**](/msit/docs/llm/工具-异常检测使用说明.md)。
   3. **单算子精度预检**: 工具提供加速库内置算子的精度预检能力，根据模型推理时 dump 的 tensor 及算子信息，计算标杆 output，再比较 dump 算子的 output 与标杆 output 的误差，以检测算子精度是否达标。可以参考 [**算子预检工具使用说明**](/msit/docs/llm/工具-精度预检使用说明.md)。

详情可以参考 [大模型精度问题定位全流程](/msit/docs/llm/大模型精度问题定位全流程.md)。

## 免责声明

- 本工具仅供调试和开发之用，不适用于生产环境。使用者需自行承担使用风险，并理解以下内容：

  - [X] 仅限调试开发使用：此工具设计用于辅助开发人员进行调试，不适用于生产环境或其他商业用途。对于因误用本工具而导致的数据丢失、损坏，本工具及其开发者不承担责任。
  - [X] 数据处理及删除：用户在使用本工具过程中产生的数据（包括但不限于dump的数据）属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防泄露或不必要的信息泄露。
  - [X] 数据保密与传播：使用者了解并同意不得将通过本工具产生的数据随意外发或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。
  - [X] 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，本工具及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。
- 在使用本工具之前，请**谨慎阅读并理解以上免责声明的内容**。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

## 历史版本安装

- 需要下载安装框架 whl 和工具 whl。
- **注意**：2024年7月起，ait仓库改名为msit。因此，7月之前的whl包仍使用ait命名，之后更新whl包会改名为msit。

- msit/ait 框架 whl:

  | 版本  | 发布日期   | 平台 | CANN 版本 | whl 链接                                                                                                                                           | MD5 校验码                       |
  | ----- | ---------- | ---- | --------- |--------------------------------------------------------------------------------------------------------------------------------------------------| -------------------------------- |
  | 0.0.1 | 2023/12/13 | arm  | 7.0.0.RC1 | [ait-0.0.1-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/ait-0.0.1-py3-none-linux_aarch64.whl) | 271051e901bb3513c7a0edbd1e096cb2 |
  | 0.0.1 | 2023/12/13 | x86  | 7.0.0.RC1 | [ait-0.0.1-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231213/ait-0.0.1-py3-none-linux_x86_64.whl)   | 9903fa06b9ff76cba667abf0cbc4da50 |
- msit_llm/ait_llm 工具 whl：

  | 版本  | 发布日期   | 平台       | CANN 版本    | whl链接                                                                                                                                                           | MD5 校验码                       | 使用指导                                                     |
  | ----- | ---------- | ---------- | ------------ |-----------------------------------------------------------------------------------------------------------------------------------------------------------------| -------------------------------- | ------------------------------------------------------------ |
  | 1.1   | 2024/05/08 | arm        | 8.0.RC2 B010 | [ait_llm-1.1-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240508/ait_llm-1.1-py3-none-linux_aarch64.whl)           |0133c8fda39ba78c2b02354b4bcf089c                                | [大模型推理精度工具](/msit/docs/llm/README.md) |
  | 1.1   | 2024/05/08 | x86        | 8.0.RC2 B010 | [ait_llm-1.1-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240508/ait_llm-1.1-py3-none-linux_x86_64.whl)            |d453b4b608b4400d77bbfb1b5c702bee                                | [大模型推理精度工具](/msit/docs/llm/README.md) |
  | 1.0   | 2024/03/22 | arm        | 8.0.RC1      | [ait_llm-1.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240325/ait_llm-1.0-py3-none-linux_aarch64.whl)          |9f7f69d49e017f98006b8191f3951868                                  | [大模型推理精度工具说明文档](/msit/docs/llm/v1.0/大模型推理精度工具说明文档.md) |
  | 1.0   | 2024/03/22 | x86        | 8.0.RC1      | [ait_llm-1.0-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240325/ait_llm-1.0-py3-none-linux_x86_64.whl)            |5a6735c9f04d3938a6384c460399ff9a                                  | [大模型推理精度工具说明文档](/msit/docs/llm/v1.0/大模型推理精度工具说明文档.md) |
  |       |            |            |              |                                                                                                                                                                 |                                  |                                                              |
  | 0.2.1 | 2024/02/08 | arm        | 8.0.RC1.B020 | [ait_llm-0.2.1-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240208/ait_llm-0.2.1-py3-none-linux_aarch64.whl)      | 1f24783f0815dbca36e8e787a8bfcf09 | [llm大模型推理精度工具功能说明_v0.2.1](/msit/docs/llm/history/llm大模型推理精度工具功能说明_v0.2.1.md) |
  | 0.2.1 | 2024/02/08 | x86        | 8.0.RC1.B020 | [ait_llm-0.2.1-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240208/ait_llm-0.2.1-py3-none-linux_x86_64.whl)        | 679fae6a5b6ea1f4a749b9554f3e5c37 | [llm大模型推理精度工具功能说明_v0.2.1](/msit/docs/llm/history/llm大模型推理精度工具功能说明_v0.2.1.md) |
  |       |            |            |              |                                                                                                                                                                 |                                  |                                                              |
  | 0.2.0 | 2024/01/17 | arm        | 8.0.RC1      | [ait_llm-0.2.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240117/ait_llm-0.2.0-py3-none-linux_aarch64.whl)      | 99b94bf7edd57b63a6e23b987d24f364 | [llm大模型推理精度工具功能说明_v0.2.0](/msit/docs/llm/history/llm大模型推理精度工具功能说明_v0.2.0.md) |
  | 0.2.0 | 2024/01/17 | x86        | 8.0.RC1      | [ait_llm-0.2.0-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20240117/ait_llm-0.2.0-py3-none-linux_x86_64.whl)        | dec5757afedfea8848c5db1bfad3d76c | [llm大模型推理精度工具功能说明_v0.2.0](/msit/docs/llm/history/llm大模型推理精度工具功能说明_v0.2.0.md) |
  |       |            |            |              |                                                                                                                                                                 |                                  |                                                              |
  | 0.1.0 | 2023/12/13 | arm, abi=0 | 7.0.0.RC1    | [ait_llm-0.1.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_aarch64.whl) | 48215f3ce18881f60beab6fad88ce30a | [llm大模型推理精度工具功能说明_v0.1.0](/msit/docs/llm/history/llm大模型推理精度工具功能说明_v0.1.0.md) |
  | 0.1.0 | 2023/12/13 | arm, abi=1 | 7.0.0.RC1    | [ait_llm-0.1.0-py3-none-linux_aarch64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI1/ait_llm-0.1.0-py3-none-linux_aarch64.whl) | b96e8e7e4786f1abcbec1458ca3ede5d | [llm大模型推理精度工具功能说明_v0.1.0](/msit/docs/llm/history/llm大模型推理精度工具功能说明_v0.1.0.md) |
  | 0.1.0 | 2023/12/13 | x86, abi=0 | 7.0.0.RC1    | [ait_llm-0.1.0-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI0/ait_llm-0.1.0-py3-none-linux_x86_64.whl)   | c605e9d50891632a09b21e90403b5b96 | [llm大模型推理精度工具功能说明_v0.1.0](/msit/docs/llm/history/llm大模型推理精度工具功能说明_v0.1.0.md) |
  | 0.1.0 | 2023/12/13 | x86, abi=1 | 7.0.0.RC1    | [ait_llm-0.1.0-py3-none-linux_x86_64.whl](https://ais-bench.obs.cn-north-4.myhuaweicloud.com/compare/20231226/ABI1/ait_llm-0.1.0-py3-none-linux_x86_64.whl)   | ea88611dc4358f51a47f7659a36d5a48 | [llm大模型推理精度工具功能说明_v0.1.0](/msit/docs/llm/history/llm大模型推理精度工具功能说明_v0.1.0.md) |
- 校验whl包是否正确

  ```
  # 校验whl包是否正确
  md5sum xxxx.whl
  ```

  比对 md5 值与所提供的校验值一致
- 安装方式：

  ```
  # 安装所需版本的框架 whl
  pip3 install ait-0.0.1-py3-none-linux_aarch64.whl
  # 安装所需版本的工具 whl
  pip3 install ait_llm-0.2.0-py3-none-linux_aarch64.whl
  ```