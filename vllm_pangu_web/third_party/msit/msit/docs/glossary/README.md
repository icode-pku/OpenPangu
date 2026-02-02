# 术语列表

#### om 文件：

- NPU上离线推理文件。类似于onnx文件，tensorflow框架的pb文件

#### atb (Ascend Transformer Boost)

- Transformer推理加速库（Ascend Transformer Boost）为实现基于Transformer的神经网络推理加速引擎库，库中包含了各类Transformer类模型的高度优化模块，如Encoder和Decoder部分。面向Transformer模型的加速库（Ascend Transformer Boost），提高Transformer模型性能，提供了基础的高性能的算子，高效的算子组合技术（Graph），方便模型加速。各类模型推理框架可以使用，目前用户有PyTorch、MindSpore、Paddle。
- 详细情况参考[《昇腾社区CANN开发套件中的开发指南对应章节》](https://www.hiascend.com/document/detail/zh/canncommercial/700/foundmodeldev/ascendtb/)

#### torchair (torch 图模式)

* torchair为用户提供了一种高效、灵活的模型部署方案，使得用户可以更加轻松地将模型应用于实际场景中。torchair将torch的FX图转换为GE计算图，并提供了GE计算图的编译与执行接口。FX图是PyTorch中的一种中间表示方式，用于表示模型的计算图和操作序列。GE计算图是昇腾AI处理器的计算图，用于表示模型的计算图和操作序列。将FX图转换为GE计算图可以实现跨平台的模型部署并加速模型的推理。
* 详细情况参考[《昇腾社区CANN开发套件中的开发指南对应章节》](https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/torchairuseguide/torchair_0002.html)
