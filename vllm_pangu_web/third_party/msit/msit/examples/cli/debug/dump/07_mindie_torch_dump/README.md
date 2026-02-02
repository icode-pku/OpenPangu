# MindIE Torch场景-整网算子数据dump

## 介绍
支持对经过MindIE Torch编译优化后的模型进行tensor数据dump。

## 1. 相关依赖

- CANN（8.0RC3及以上）
- MindIE（1.0RC3及以上）

## 2. msit安装

- msit安装请参考[一体化安装指导](../../../../../docs/install/README.md)
- 确保安装msit下面的compare和llm组件
  ```sh
  msit install compare llm
  ```

## 3. Dump 数据 

在精度比对之前，首先需要分别dump CPU和NPU数据，下面以ResNet50模型为例，提供详细脚本给大家参考。

### 3.1 NPU数据Dump（TorchScript路线）

- 准备 `resnet_inference.py`，为模型的前向推理脚本, 目前MindIE Torch支持TS和export路线的数据dump。
- 由于MindIE Torch的推理接口是异步接口，为了保证推理完成从而实现推理过程数据落盘，需要进行同步操作。目前MindIE Torch和torch_npu提供了两套不同的接口，同步操作方法如下：
  ```python
  # torch_npu 接口   推荐使用！！
  import torch_npu
  npu_result = compiled_module(input_data.to("npu:0"))
  torch_npu.npu.synchronize()
  mindietorch.finalize()
  # 环境中未安装torch_npu依赖包时，支持使用MindIE Torch 接口, 使用方式如下:
  npu_stream = mindietorch.npu.Steam()
  npu_result = compiled_module(input_data.to("npu:0"))
  with mindietorch.npu.stream(npu_stream):
    npu_stream.synchronize()
  ```

#### 3.1.1 TorchScript路线 (jit.ScriptModule)
  ```python
  # 请务必先导入torch，再导入mindietorch
  import torch
  import mindietorch
  import torchvision.models as models

  model = models.resnet50()
  model.eval()
  torch.manual_seed(88)   # 设置随机种子，确保cpu和npu侧模型输入一致
  input_data = torch.randn(1, 3, 224, 224)
  
  input_info = [mindietorch.Input((1, 3, 224, 224))]
  mindietorch.set_device(0)  # 指定推理卡

  traced_model = torch.jit.trace(model, input_data)
  compiled_module = mindietorch.compile(traced_model, inputs=input_info, soc_version={具体的芯片型号}) # 芯片型号可以通过npu-smi info查看得到
  # 数据同步
  import torch_npu
  npu_result = compiled_module(input_data.to("npu:0"))
  torch_npu.npu.synchronize()
  
  mindietorch.finalize()
  ```
#### 3.1.2 Torch.export路线 (export.ExportedProgram 或 nn.Module)
  ```python
  import torch
  from torch._export import export
  import mindietorch
  import torchvision.models as models

  model = models.resnet50()
  model.eval()
  input_data = torch.randn(1, 3, 224, 224)

  input_info = [mindietorch.Input((1, 3, 224, 224))]
  model_ep = export(model, args=(input_data,))  # 不使用export即为nn.Module模型
  compiled_module = mindietorch.compile(model_ep, inputs=input_info, soc_version={具体的芯片型号}, ir="dynamo")    # 必须指定ir为dynamo

  mindietorch.set_device(0)
  import torch_npu
  result = compiled_module(input_data.to("npu:0"))
  torch_npu.npu.synchronize()
  mindietorch.finalize()
  ```
#### 3.1.3 Torch.export路线 （fx.GraphModule）
  ```python
  import torch
  from torch._export import export
  from torch import fx
  import mindietorch
  import torchvision.models as models

  def fx_transform(m: torch.nn.Module, tracer_class=fx.Tracer):
      graph = tracer_class().trace(m)
      graph.lint()
      new_gm = fx.GraphModule(m, graph)
      return new_gm
  
  input_data = torch.randn(1, 3, 224, 224)
  model = models.resnet50()
  model.eval()
  fx_model = fx_transform(model)
  
  inputs = [mindietorch.Input((1, 3, 224, 224))] 
  compiled_module = mindietorch.compile(fx_model, inputs=inputs, soc_version={具体的芯片型号}, ir="dynamo") # 必须指定ir为dynamo

  mindietorch.set_device(0)
  import torch_npu
  npu_result = compiled_module(input_data.to("npu:0"))
  torch_npu.npu.synchronize()
  mindietorch.finalize()
  ```


- 根据自己的场景编辑完resnet_inference.py后，通过msit命令Dump NPU数据

  ```sh
  msit debug dump --exec "python resnet_inference.py" [--output /path/to/dump] [--operation-name MatMulv2_1,trans_Cast_0]
  ```

- 参数说明 

| 参数名          | 描述                                                                                                                             | 必选 |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ---- |
| --exec | MindIE Torch推理脚本的执行命令                                                                                   | 是   |
| --output | 指定Dump数据输出路径，默认为当前路径                                 | 否   | 
| -opname, --operation-name | 需要Dump的算子，默认为 all，表示会对模型中所有 op 进行 Dump，其中元素为MindIE Torch算子类型，若设置 operation-name，只会 Dump 指定的 op                                | 否   | 

### 3.2 标杆数据Dump

- 精度比对的标杆数据一般选取的是Torch单算子模式下推理数据，具体Dump流程请参考[PyTorch 场景的精度数据采集](../../../../../docs/llm/工具-Pytorch场景数据dump.md)

## 注意

- NPU数据Dump时请在线进行推理，若save模型后load再进行推理会使得json文件缺少必要信息，无法进行compare。

- NPU数据Dump后会在当前路径下生成两个算子类型映射关系文件，名称为：`mindie_torch_op_mapping.json` 和 `mindie_rt_op_mapping.json` 。

- 由于transformers库内部会导入torch_npu，因此如果导入了transformers库且环境中安装了torch_npu的场景下必须使用torch_npu的同步接口实现推理过程数据落盘，否则会报错。