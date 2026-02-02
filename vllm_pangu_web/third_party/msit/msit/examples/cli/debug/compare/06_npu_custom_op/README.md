# Npu Custom op


## 介绍

某些昇腾模型，存在NPU自定义算子，比如 [Retinanet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet)，存在BatchMultiClassNMS后处理算子，该算子无法在onnxruntime上运行，导致该模型无法使用msit debug compare功能进行精度比对。这时可以通过添加--custom-op参数，指定自定义算子类型名称，工具会将这些算子进行删除使其能够正常推理并获取dump数据。

## 使用场景约束

1、只支持标杆模型为onnx文件，[-gm，--golden-model]入参必须为.onnx模型；

2、当前custom-op取值范围："BatchMultiClassNMS"、"DeformableConv2D"、"RoiExtractor"

3、使用时请勿关闭--dump，不要开启--locat、--single-op等高级功能


## 运行原理

- 将原模型中存在的自定义算子，全部删除，然后把自定义算子输出节点，添加到整体模型的inputs节点中；

- 在npu上，运行om模型，获取npu dump数据；

- 通过npu dump数据，获取npu自定义算子的输出数据，传入onnx模型；

- 在cpu上运行删除了自定义算子的onnx模型，获取标杆dump数据；

- 调用CANN包里的精度比对工具，比对onnx的dump数据和npu dump数据。

## 运行示例

以[Retinanet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/contrib/cv/detection/Retinanet)模型为例，获取到onnx模型后，查看模型中自定义算子类型为"BatchMultiClassNMS"，则命令如下：

  ```sh
  msit debug compare -gm ./model.onnx -om ./model.om -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test  --custom-op='BatchMultiClassNMS'
  ```
  - `--custom-op` 为onnx模型中自定义算子类型名称

也支持多个自定义算子类型，中间用英文逗号隔开：
  ```sh
  msit debug compare -gm ./model.onnx -om ./model.om -c /usr/local/Ascend/ascend-toolkit/latest -o /home/HwHiAiUser/result/test  --custom-op='BatchMultiClassNMS,RoiExtractor'
  ```
