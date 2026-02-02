# LLM 迁移分析
- [PyTorch transformers LLM 模型迁移生成 atb 浮点模型](#pytorch-transformers-llm-模型迁移生成-atb-浮点模型)
  - [限定条件](#限定条件)
  - [环境说明](#环境说明)
  - [参数说明](#参数说明)
- [ATB python 迁移示例](#atb-python-迁移示例)
  - [Transformers LLaMA 迁移到 ATB python 模型](#transformers-llama-迁移到-atb-python-模型)
  - [Transformers LLaMA 迁移到 ATB python 量化模型](#transformers-llama-迁移到-atb-python-量化模型)
- [ATB cpp 迁移示例](#atb-cpp-迁移示例)
  - [Transformers QWEN 13B 迁移到 ATB cpp 示例](#transformers-qwen-13b-迁移到-atb-cpp-示例)
  - [ATB cpp 仅生成 python 调用代码](#atb-cpp-仅生成-python-调用代码)
- [llm 浮点模型 layer 层 cpp 稀疏量化迁移](#llm-浮点模型-layer-层-cpp-稀疏量化迁移)
  - [准备](#准备)
  - [Baichuan2 7B 迁移示例](#baichuan2-7b-迁移示例)
- [输出大模型迁移分析报告](#输出大模型迁移分析报告)
***

## PyTorch transformers LLM 模型迁移生成 atb 浮点模型
- 由 PyTorch transformers LLM 模型迁移生成 atb 浮点模型，包括 model、layer 层代码，以及相应的 cpp 与 h 文件
- 迁移后需要基于加速库 ATB 实现，因此支持的 oprations 限定在 ATB 已有算子

### 限定条件
- **适用于 transformers 包，支持类似 LLaMA、QWEN 的典型 LLM 模型结构迁移，以及 LLaVA 等 VL 模型迁移**
- **当前 MindIE python 接口发布包基于 python 3.10/ python 3.11，迁移功能也限定 python3.10/ python 3.11；且 transformers 版本需要支持对应模型的 FX 构图，即 `transformers.utils.fx.symbolic_trace` 接口**
- **使用最新的 mindIE B032 版本需要适配transformers==4.45.2**
- **ATB Python 模型当前硬件限定 Atlas 800I A2 / 800T A2 / 900 A2 / 300I / 300I Pro / 300I Duo** 
- **ATB C++ 模型迁移适配 ATB RC3.B030 + mindie 1.0.RC3.B030**
- **在使用迁移功能时限制只允许从本地导入权重及配置文件**
- **需自行保证原始模型、权重及配置文件安全可靠。因导入不安全文件而导致的任何安全风险或损失由用户自行承担**

### 环境说明
- 安装 msit
- ATB cpp 模型迁移需要获取 [Gitcode ascend/MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM) 最新源码
- ATB Python 模型迁移需要安装 MindIE
- 准备待迁移的 transformers 模型目录


### 参数说明
```sh
msit llm transform [-h] -s SOURCE [-atb ATB_MODEL_PATH] [--enable-sparse] [--to-python] [--to-quant]
                   [--quant-disable-names QUANT_DISABLE_NAMES] [-a] [-l {debug,info,warning,error,fatal,critical}]
```

| 参数名                 | 描述                                                                                                                                                                | 必选 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| -s, --source           | 指定待迁移的代码路径，支持指定文件夹，将迁移文件夹下所有不包含 `quant` 的文件；或指定单独的 cpp 文件，将迁移该 cpp 文件，以及同名的 h 文件                          | 是   |
| -a, --analyze          | 指定是否需要输出大模型迁移分析报告，不指定则不输出                                                                                                                  | 否   |
| -atb, --atb_model_path | 指定待调用的 cpp 代码路径。支持指定文件夹，参数指定为目录时目录下必须只存在一个cpp文件和一个h文件；或指定为文件，文件同级目录下必须存在同文件名的 cpp 文件和 h 文件 | 否   |
| --enable-sparse        | 指定迁移为稀疏量化模型，不指定则迁移为量化模型                                                                                                                      | 否   |
| --to-python, -py       | 指定在 Torch 模型迁移到 ATB 模型场景下，迁移为 ATB python 接口模型                                                                                                  | 否   |
| --to-quant, -quant     | 指定在 Torch 模型迁移到 ATB python 接口模型场景下，迁移为量化模型，**需要配合 `--to-python` 使用**                                                                      | 否   |
| --quant-disable-names  | 文件或 ',' 分割的字符串，指定在 Torch 模型迁移到 ATB python 接口量化模型场景下，量化回退层的名称；**需要配合 `--to-python` 与 `--to-quant` 使用**；默认值 None 表示回退 `lm_head` 层 | 否   |
| -l, --log-level        | 指定log level，默认为 info，可选值 debug, info, warning, error, fatal, critical                                                                                     | 否   |
| -h, --help             | 命令行参数帮助信息|  否 |
***

## ATB python 迁移示例
### Transformers LLaMA 迁移到 ATB python 模型
- 从 huggingface 获取相应 LLaMA 模型
- **迁移生成 ATB python 浮点模型**，将生成迁移完成的 ATB python 模型代码 py 文件，以及模型配置参数，并给出调用示例
- 如果迁移报错，可以使用 export ASDOPS_LOG_LEVEL=INFO，export ASDOPS_LOG_TO_STDOUT=1 打印日志查看报错信息
  ```sh
  msit llm transform -s test_llama/ -py
  # ...
  # ==============================
  # Saved to: llamaforcausallm_atb_float.py
  #
  # ==============================
  # atb_model config:
  # {'vocab_size': 32000, 'num_attention_heads': 32, 'head_dim': 32, 'max_batch_size': 1, 'max_seq_len': 1024}
  #
  # ==============================
  # Run like:
  #
  # python3 -c "
  # import torch, torch_npu
  # import llamaforcausallm_atb_float
  # from msit_llm.transform.torch_to_atb_python import ATBModel
  # 
  # atb_model = ATBModel(llamaforcausallm_atb_float.Model())
  # weights = torch.load('$WEIGHT_PATH')  # Use actual WEIGHT_PATH
  # atb_model.set_weights(weights)
  # 
  # input_len = 32
  # out = atb_model.forward(input_ids=torch.arange(input_len),position_ids=torch.arange(input_len))
  # print(out)
  # "
  #
  # ==============================
  # End-to-end inference example saved to: run.py
  # Execute by: python run.py
  ```
  参照输出的 `Run like:` 部分，导入生成的 py 文件，并调用推理
  ```py
  import torch, torch_npu
  import llamaforcausallm_atb_float
  from msit_llm.transform.torch_to_atb_python import ATBModel

  atb_model = ATBModel(llamaforcausallm_atb_float.Model())
  weights = torch.load('test_llama/state_dict.pt')  # Use actual WEIGHT_PATH
  atb_model.set_weights(weights)

  input_len = 32
  out = atb_model.forward(input_ids=torch.arange(input_len),position_ids=torch.arange(input_len))
  print({kk: vv.shape for kk, vv in out.items()})
  # {'output': torch.Size([32, 32000])}
  ```
  也可直接运行 python run.py 调用推理
  ```sh
  python run.py

  # ==============================
  # Input: 好雨知时节，当春
  # Output: 好雨知时节，当春乃发生。
  ```
    | 参数名          | 描述                                                                                                                                                                | 必选 |
  |--------------| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |----|
  | -i, --inputs | 多模态模型推理中需要的对于输入图片的文字描述，默认值为"Who's there?"                            | 否  |
  | -w, --weight | 多模态模型的权重路径，默认使用输入的模型路径中的浮点权重| 否  |
  | -h, --help   | 命令行参数帮助信息| 否  |
### Transformers LLaMA 迁移到 ATB python 量化模型
- 从 huggingface 获取相应 LLaMA 模型
- **需要相应量化权重**
- **迁移生成 ATB python 量化模型**，将生成迁移完成的 ATB python 量化模型代码 py 文件，以及模型配置参数，并给出调用示例
  ```sh
  msit llm transform -s test_llama/ -py -quant
  # ...
  # calling convert_to_quant, quant_disable_names = ['lm_head']
  # ==============================
  # Saved to: llamaforcausallm_atb_quant.py
  #
  # ==============================
  # atb_model config:
  # {'vocab_size': 32000, 'num_attention_heads': 32, 'head_dim': 32, 'max_batch_size': 1, 'max_seq_len': 1024}
  #
  # ==============================
  # Run like:
  #
  # python3 -c "
  # import torch, torch_npu
  # import llamaforcausallm_atb_quant
  # from msit_llm.transform.torch_to_atb_python import ATBModel
  # 
  # atb_model = ATBModel(llamaforcausallm_atb_quant.Model())
  # weights = torch.load('$WEIGHT_PATH')  # Use actual WEIGHT_PATH
  # atb_model.set_weights(weights)
  # 
  # input_len = 32
  # out = atb_model.forward(input_ids=torch.arange(input_len),position_ids=torch.arange(input_len))
  # print(out)
  # "
  ```
  参照输出的 `Run like:` 部分，导入生成的 py 文件，并调用推理
  ```py
  import torch, torch_npu
  import llamaforcausallm_atb_quant
  from msit_llm.transform.torch_to_atb_python import ATBModel

  atb_model = ATBModel(llamaforcausallm_atb_quant.Model())
  weights = torch.load('test_llama/quant_state_dict.pt')  # Use actual WEIGHT_PATH
  atb_model.set_weights(weights)

  input_len = 32
  out = atb_model.forward(input_ids=torch.arange(input_len),position_ids=torch.arange(input_len))
  print({kk: vv.shape for kk, vv in out.items()})
  # {'output': torch.Size([32, 32000])}
  ```

### Transformers LLava 迁移到 ATB python 模型
- 从 huggingface 获取相应 LLava 模型  
（注：transformers==4.44.2）
- **迁移生成 ATB python 浮点模型**，将生成迁移完成的 ATB python 模型代码 py 文件，以及模型配置参数，并给出调用示例
  ```sh
  msit llm transform -s test_llava/ -py
  # ...
  # ==============================
  # Saved to: llamaforcausallm_atb_float.py
  #
  # ==============================
  # atb_model config:
  # {'vocab_size': 32000, 'num_attention_heads': 32, 'head_dim': 32, 'max_batch_size': 1, 'max_seq_len': 1024}
  #
  # ==============================
  # Run like:
  #
  # python3 -c "
  # import torch, torch_npu
  # import llamaforcausallm_atb_float
  # from msit_llm.transform.torch_to_atb_python import ATBModel
  # 
  # atb_model = ATBModel(llamaforcausallm_atb_float.Model())
  # weights = torch.load('$WEIGHT_PATH')  # Use actual WEIGHT_PATH
  # atb_model.set_weights(weights)
  # 
  # input_len = 32
  # out = atb_model.forward(input_ids=torch.arange(input_len),position_ids=torch.arange(input_len))
  # print(out)
  # "
  #
  # ==============================
  # End-to-end inference example saved to: run_vl.py
  # Execute by: python run_vl.py
  ```
  可直接运行 python run_vl.py 调用推理
  ```sh
  python run_vl.py -i {image_path} -t "text"
  ```
  
  | 参数名                 | 描述                                                                                                                                                                | 必选 |
  | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
  | -i, --image           | 多模态模型推理中需要的图片路径                          | 是   |
  | -t, --text          | 多模态模型推理中需要的对于输入图片的文字描述，默认值为"Describe the image."                                                                                                           | 否   |
  | -w, --weight             | 多模态模型的权重路径，默认使用输入的模型路径中的浮点权重，可更改为量化后权重路径|  否 |
  | -h, --help             | 命令行参数帮助信息|  否 |
***

## ATB cpp 迁移示例
### Transformers QWEN 13B 迁移到 ATB cpp 示例
- **Step 1. 迁移生成 atb 模型代码**，`--source` 指定待迁移的 transformers 模型目录，生成迁移后 model 以及 layer 的 cpp 与 h 代码，同时生成 python 代码，用于推理时调用。
  ```sh
  msit llm transform -s /data/qwen-14b-chat
  # Generated files: [
  #     qwenlmheadmodel/model/decoder_model.cpp,
  #     qwenlmheadmodel/model/decoder_model.h,
  #     qwenlmheadmodel/layer/decoder_model.cpp,
  #     qwenlmheadmodel/layer/decoder_model.h,

  #     qwen/run.py,
  #     qwen/router_qwen.py,
  #     qwen/modeling_qwen.py,
  #     qwen/flash_causal_qwen.py,
  # ]
  ```
- **Step 2.** 将生成的 cpp 和 h 代码放到 `MindIE-LLM` 模型目录下（python 代码无需复制），**该路径基于不同的 MindIE 版本可能会不同**
  ```sh
  mv qwenlmheadmodel ~/MindIE-LLM/examples/atb_models/atb_framework/models
  ```
- **Step 3.** 重新编译 `MindIE-LLM`，**实际的编译路径与命令需要参照 MindIE 文档**
  ```sh
  cd ~/MindIE-LLM/examples/atb_models
  bash scripts/build.sh
  ```
  由于迁移的适配性问题，以及 `MindIE-LLM` 迭代更新，编译过程可能存在报错，仍依赖用户手动修复错误。
- **Step 4.** 运行 run.py，执行推理，--model_path 指定 transformers 模型目录
  ```sh
  python qwen/run.py --model_path=/data/qwen-14b-chat
  ```
  run.py的参数说明见[MindIE-LLM中run_pa.py的参数说明](https://gitcode.com/Ascend/MindIE-LLM/blob/master/examples/atb_models/examples/README.md#run_papy%E8%84%9A%E6%9C%AC%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D)

  由于迁移的适配性问题，以及 `MindIE-LLM` 迭代更新，推理过程可能存在报错，仍依赖用户手动修复 python 文件中错误。
### ATB cpp 仅生成 python 调用代码
- 若已存在 model 的 cpp 和 h 代码，可通过指定 -atb 或 --atb_model_path 来生成 python 调用代码

  ```sh
  msit llm transform -s /data/qwen-14b-chat -atb ~/MindIE-LLM/examples/atb_models/atb_framework/models/qwen/model/decoder.cpp

  # -atb 参数指定为目录时目录下必须只存在一个cpp文件和一个h文件，否则无法自动识别
  #      指定为文件时，文件同级目录下必须存在同文件名的cpp文件和h文件
  # -atb ~/MindIE-LLM/examples/atb_models/atb_framework/models/qwen/model/decoder.cpp
  # -atb ~/MindIE-LLM/examples/atb_models/atb_framework/models/qwen/model/decoder.h
  # -atb qwenlmheadmodel/model
  # -atb qwenlmheadmodel/model/decoder.cpp

  # Generated files: [
  #     qwen/run.py,
  #     qwen/router_qwen.py,
  #     qwen/modeling_qwen.py,
  #     qwen/flash_causal_qwen.py,
  # ]
  ```
***

## llm 浮点模型 layer 层 cpp 稀疏量化迁移
- 由 llm 的浮点模型 layer 层代码，迁移生成稀疏量化 layer 层代码，包括 cpp 文件与 h 文件

### 准备
- 安装 msit
- 获取 [Gitcode ascend/MindIE-LLM](https://gitcode.com/Ascend/MindIE-LLM) 源码，找到待迁移模型 layer 定义

### Baichuan2 7B 迁移示例
- 模型 layer 定义位置
  ```sh
  cd ModelLink/mindie_ref/mindie_llm/atb_models/models/baichuan2/7b  # layer 定义位置
  ls
  # layer  model  operation
  ```
- 迁移为量化模型
  ```sh
  msit llm transform -s layer/flash_attention_rope_layer.cpp
  # ...
  # Transformed source files: [
  #     layer/flash_attention_rope_layer.cpp
  #     layer/flash_attention_rope_layer.h
  # ]
  # Transformed target files: [
  #     layer/quant_flash_attention_rope_layer.cpp
  #     layer/quant_flash_attention_rope_layer.h
  # ]
  ```
  - 由 `layer/flash_attention_rope_layer.cpp` `layer/flash_attention_rope_layer.h` 迁移生成 `layer/quant_flash_attention_rope_layer.cpp` `layer/quant_flash_attention_rope_layer.h`
  - 其中 `FlashAttentionRopeLayerTensorId` 节点中增加了 `_DESCALE` `_BIAS` 相关节点，同时 `RmsNorm` `Linear` `MLP` 等相关节点更新了属性及输入 tensor 节点
- 指定 `--enable-sparse` 迁移为稀疏量化模型
  ```sh
  msit llm transform -s layer/flash_attention_rope_layer.cpp --enable-sparse
  # ...
  # Transformed source files: [
  #     layer/flash_attention_rope_layer.cpp
  #     layer/flash_attention_rope_layer.h
  # ]
  # Transformed target files: [
  #     layer/sparse_quant_flash_attention_rope_layer.cpp
  #     layer/sparse_quant_flash_attention_rope_layer.h
  # ]
  ```
  - 由 `layer/flash_attention_rope_layer.cpp` `layer/flash_attention_rope_layer.h` 迁移生成 `layer/sparse_quant_flash_attention_rope_layer.cpp` `layer/sparse_quant_flash_attention_rope_layer.h`
  - 其中 `LinearParam` 更新为 `LinearSparseParam`，并在 `FlashAttentionRopeLayerTensorId` 增加了 `_INDEX` 相关节点
***

## 输出大模型迁移分析报告
- 在大模型迁移时，工具支持输出大模型使用DAG获取的更加细粒度的模型结构json文件，及DAG模型算子是否能获取对应的加速库算子以支持大模型迁移的算子支持度csv文件。
- 执行示例
  ```shell
  msit llm transform -s {SOURCE} -a
  ```
