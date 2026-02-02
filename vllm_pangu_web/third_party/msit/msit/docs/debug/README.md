### msit debug一站式调试组件简介
提供一站式调试功能，用于传统小模型下定位用户推理过程中的问题，确保模型的正确性。该模块包括：

#### Dump
提供了传统小模型场景下的数据 dump 功能，适用于TensorFlow、TensorFlow 2.0、ONNX、Caffe、MindIE-Torch框架。

[dump快速入门指南](./dump/README.md) 

#### Compare
提供了传统小模型推理场景下的自动化比对功能，用于定位问题算子，适用于TensorFlow、TensorFlow 2.0、ONNX、Caffe、MindIE-Torch框架。

 [compare快速入门指南](./compare/README.md)

#### OpCheck
提供了传统小模型场景下精度预检功能，支持对经过GE推理后 dump 落盘数据进行算子精度预检，检测kernel级别的算子精度是否达标。\
**注:** 目前只支持TensorFlow 2.6.5

#### Surgeon
使能ONNX模型在昇腾芯片的优化，并提供基于ONNX的改图功能。

[surgeon快速入门指南](./surgeon/README.md)