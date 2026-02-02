# Specify Input Shape Info


## 介绍

指定模型输入的shape信息(动态场景必须进行指定)。

## 运行示例
**注意**：
- 当输入存在类似scalar向量（shape为()时），不需要指定其shape，直接跳过即可。
- 目前该功能只支持由atc的input_shape_range参数转化而来的om模型。
------------------------------------------------

1. 指定-is或--input-shape进行精度对比。
  ```sh
  msit debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -is "image:1,3,224,224"
  ```
如果模型为动态shape模型，则会以该-is输入的shape信息进行模型推理和精度对比。

2. 指定-dr或--dym-shape-range进行动态模型多个shape情况的精度对比。(优先级比-is,--input-shape更高)
  ```sh
  msit debug compare -gm /home/HwHiAiUser/onnx_prouce_data/resnet_offical.onnx -om /home/HwHiAiUser/onnx_prouce_data/model/resnet50.om \
  -dr "image:1,3,224-256,224~226"
  ```
- 其中，input_name必须是转换前的网络模型中的节点名称；"\~"表示范围，a\~b\~c含义为[a: b :c]；"-"表示某一位的取值。
以上总共会进行6次精度对比流程，分别对输入为["image:1,3,224,224","image:1,3,224,225","image:1,3,224,226","image:1,3,256,224","image:1,3,256,225","image:1,3,256,226"]的情况进行比较。
