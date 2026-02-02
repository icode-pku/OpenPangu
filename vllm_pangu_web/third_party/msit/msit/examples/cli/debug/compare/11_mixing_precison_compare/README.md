# 混合策略精度比对

## 介绍
使用ATC工具在转换离线模型时可以选择不同的浮点精度模式。为了排查在不同精度模式下算子产生的精度问题，可以直接将不同精度模式下产生的.om模型进行精度比对。

在ATC转换中设置模型的浮点精度具体可以参照[ATC工具使用指南--precision_mode_v2](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/atc/atlasatcparam_16_0069.html)。


## 运行示例
使用ATC对.onnx模型转换为不同精度策略的.om模型（以场景一：force_fp32与场景二：force_fp16例子）：
```
atc --framework=5 --soc_version=<soc_version> --model=./resnet50.onnx --output=./resnet50_force_fp32 --precision_mode=force_fp32

atc --framework=5 --soc_version=<soc_version> --model=./resnet50.onnx --output=./resnet50_force_fp16 --precision_mode=force_fp16

```
将生成的.om模型放在当前目录下，执行命令如下（注意：要求-gm后使用的.om模型为融合程度较低的场景一策略作为标杆数据，即force_fp32；-om后使用的.om模型为使用融合程度较高的场景二策略，即force_fp16)：
```
msit debug compare -gm {fp32_om_model_path} -om {fp16_om_model_path} -o {output_file_path} 
```

## 结果

- Tensor比对结果result_*.csv文件路径，参考[对比结果分析步骤](../result_analyse/README.md)。

- csv表格中输出参数和onnx比对om模型有区别，其中[GroundTruth]和[NPUDump]表示的场景一与场景二下的om离线模型算子名：


  | 参数          | 说明                       |
  |-------------| ---------------------- |
  | GroundTruth | 表示场景一下精度策略的离线模型的算子名。 | 
  | NPUDump     | 表示场景二下精度策略的离线模型的算子名。| 